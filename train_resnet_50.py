# %%
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import nn
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import os
import datetime
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# %%
# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return torch.mean(F_loss)
        elif self.reduction == "sum":
            return torch.sum(F_loss)
        else:
            return F_loss


# %%
PATH = "/home/ubuntu/Final-Project-Group4"
EXCEL_PATH = PATH + os.path.sep + "dataset" + os.path.sep + "final_dataset.xlsx"
DATA_DIR = PATH + os.path.sep + "dataset" + os.path.sep + "train" + os.path.sep

# %%
df = pd.read_excel(EXCEL_PATH)
# One-hot encode the 'Category' column
one_hot_encoded = df["Category"].str.get_dummies(sep=",")

# Convert the one-hot encoded DataFrame to a string of 0s and 1s separated by commas for each row
df["target_class"] = one_hot_encoded.apply(lambda x: ",".join(x.astype(str)), axis=1)
train_data, test_data = train_test_split(df, test_size=0.20, random_state=42)
# %%


class MultiLabelImageDataset(Dataset):
    def __init__(self, list_IDs, type_data, transform=None):
        self.type_data = type_data
        self.list_IDs = list_IDs
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        if self.type_data == "train":
            y = train_data.target_class.get(ID)
            file = DATA_DIR + train_data.ImageId.get(ID)
        else:
            y = test_data.target_class.get(ID)
            file = DATA_DIR + test_data.ImageId.get(ID)

        y = y.split(",")
        labels_ohe = [int(e) for e in y]
        y = torch.FloatTensor(labels_ohe)

        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return img, y


# %%

train_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = MultiLabelImageDataset(
    list_IDs=train_data.index,
    type_data="train",
    transform=train_transform,
)
test_dataset = MultiLabelImageDataset(
    list_IDs=test_data.index,
    type_data="test",
    transform=test_transform,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# %%
label_matrix = train_data["Category"].str.get_dummies(",")
class_frequencies = label_matrix.sum()
total_samples = class_frequencies.sum()
class_weights = total_samples / (class_frequencies * len(class_frequencies))
class_weight_tensor = torch.tensor(class_weights.values, dtype=torch.float)
sample_weights = label_matrix.dot(class_weight_tensor).values

# %%
sampler = WeightedRandomSampler(
    sample_weights, num_samples=len(train_dataset), replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=512, sampler=sampler, num_workers=6)
validation_loader = DataLoader(test_dataset, batch_size=512, num_workers=6)

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_classes = 46

# %%
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)
model.fc.sigmoid = nn.Sigmoid()
model.fc.requires_grad = True

criterion = FocalLoss(alpha=1, gamma=2)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.1, patience=2, verbose=True
)

num_epochs = 10
unfreeze_epoch = 2
model.to(device)
best_val_f1_macro = 0

train_losses = []
val_losses = []
val_f1_micros = []
val_f1_macros = []

# %%

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    steps_train = 0.0
    if epoch >= unfreeze_epoch:
        for param in model.layer4.parameters():
            param.requires_grad = True

    if epoch >= unfreeze_epoch + 2:
        for param in model.layer3.parameters():
            param.requires_grad = True

    if epoch >= unfreeze_epoch + 4:
        for param in model.layer2.parameters():
            param.requires_grad = True
    with tqdm(total=len(train_loader), desc="Epoch {}".format(epoch)) as pbar:
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            steps_train += 1
            train_loss += loss.item() * images.size(0)
            pbar.update(1)
            pbar.set_postfix_str("Train Loss: {:.5f}".format(train_loss / steps_train))
    train_loss /= len(train_dataset)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    steps_test = 0.0
    predictions = []
    true_labels = []

    with torch.no_grad():
        with tqdm(total=len(validation_loader), desc="Epoch {}".format(epoch)) as pbar:
            for images, labels in validation_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                predictions.extend(outputs.sigmoid().cpu().numpy() >= 0.5)
                true_labels.extend(labels.cpu().numpy())
                steps_test += 1
                pbar.update(1)
                pbar.set_postfix_str("Test Loss: {:.5f}".format(val_loss / steps_test))

    val_loss /= len(test_dataset)
    val_losses.append(val_loss)

    f1_micro = f1_score(true_labels, predictions, average="micro")
    f1_macro = f1_score(true_labels, predictions, average="macro")
    f1_scores = f1_score(true_labels, predictions, average=None)
    val_accuracy_score = accuracy_score(true_labels, predictions)
    val_f1_micros.append(f1_micro)
    val_f1_macros.append(f1_macro)

    print(
        f"Epoch [{epoch+1}/{num_epochs}], "
        f"Train Loss: {train_loss:.4f}, "
        f"Val Loss: {val_loss:.4f}, "
        f"Val F1 Micro: {f1_micro:.4f}, "
        f"Val F1 Macro: {f1_macro:.4f}, "
        f"Val Accuracy: {val_accuracy_score:.4f}"
    )
    print("Class-wise F1 Scores:")
    for i, score in enumerate(f1_scores):
        print(f"Class {i+1}: {score:.4f}")

    if f1_macro > best_val_f1_macro:
        best_val_f1_macro = f1_macro
        best_epoch = epoch + 1
        torch.save(
            model.state_dict(), f"best_model_{datetime.datetime.now().isoformat()}.pt"
        )

    print(classification_report(true_labels, predictions))
    scheduler.step(f1_macro)

print(f"Best Validation F1 Macro: {best_val_f1_macro:.4f} (Epoch {best_epoch})")

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%
