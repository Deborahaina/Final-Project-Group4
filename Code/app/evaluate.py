from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
import pandas as pd
import numpy as np
import cv2
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
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

category_mapping = {
    0: "shirt, blouse",
    1: "top, t-shirt, sweatshirt",
    2: "dress",
    3: "jumpsuit",
    4: "cape",
    5: "glasses",
    6: "hat",
    7: "headband, head covering, hair accessory",
    8: "tie",
    9: "glove",
    10: "watch",
    11: "belt",
    12: "sweater",
    13: "leg warmer",
    14: "tights, stockings",
    15: "sock",
    16: "shoe",
    17: "bag, wallet",
    18: "scarf",
    19: "umbrella",
    20: "hood",
    21: "collar",
    22: "lapel",
    23: "cardigan",
    24: "epaulette",
    25: "sleeve",
    26: "pocket",
    27: "neckline",
    28: "buckle",
    29: "zipper",
    30: "applique",
    31: "bead",
    32: "bow",
    33: "flower",
    34: "jacket",
    35: "fringe",
    36: "ribbon",
    37: "rivet",
    38: "ruffle",
    39: "sequin",
    40: "tassel",
    41: "vest",
    42: "pants",
    43: "shorts",
    44: "skirt",
    45: "coat",
}


results = {}


cnn_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        x = self.pool(self.act(self.bn2(self.conv2(x))))
        x = self.pool(self.act(self.bn3(self.conv3(x))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def load_model(model_name, model_path):
    if model_name == "ViT_B_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        model.heads.head = nn.Linear(
            model.heads.head.in_features, len(category_mapping)
        )
        model.heads.head.sigmoid = nn.Sigmoid()
    elif model_name == "ResNet_101":
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(category_mapping))
        model.fc.sigmoid = nn.Sigmoid()
    elif model_name == "CNN":
        model = CNN(len(category_mapping))

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model


def load_and_evaluate_model(model_name, model_path, data_loader, device):
    model = load_model(model_name, model_path)
    accuracy, f1, precision, recall, class_report = evaluate_model(
        model, data_loader, device
    )
    return accuracy, f1, precision, recall, class_report


# %%
PATH = "/home/ubuntu/Final-Project-Group4"
EXCEL_PATH = PATH + os.path.sep + "dataset" + os.path.sep + "final_dataset.xlsx"
DATA_DIR = PATH + os.path.sep + "dataset" + os.path.sep + "train" + os.path.sep

# %%
df = pd.read_excel(EXCEL_PATH)
one_hot_encoded = df["Category"].str.get_dummies(sep=",")
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
        img = Image.fromarray(img)  # Convert numpy array to PIL Image

        if self.transform:
            img = self.transform(img)

        return img, y


# %%
def evaluate_model(model, data_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted = torch.sigmoid(outputs).round()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="micro")
    precision = precision_score(y_true, y_pred, average="micro")
    recall = recall_score(y_true, y_pred, average="micro")

    class_report = classification_report(
        y_true, y_pred, target_names=list(category_mapping.values()), output_dict=True
    )

    return accuracy, f1, precision, recall, class_report


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# %%

for model_name in ["ViT_B_16", "ResNet_101", "CNN"]:
    if model_name == "ViT_B_16":
        test_transform = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
    elif model_name == "ResNet_101":
        test_transform = models.ResNet101_Weights.DEFAULT.transforms()
    elif model_name == "CNN":
        test_transform = cnn_transform
    test_dataset = MultiLabelImageDataset(
        list_IDs=test_data.index,
        type_data="test",
        transform=test_transform,
    )
    validation_loader = DataLoader(test_dataset, batch_size=64, num_workers=4)
    model_path = f"model_{model_name.lower()}.pt"
    accuracy, f1, precision, recall, class_report = load_and_evaluate_model(
        model_name, model_path, validation_loader, device
    )
    results[model_name] = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        "Class Report": class_report,
    }
    print(model_name)
    print(
        {
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
            "Class Report": class_report,
        }
    )

# %%

import pandas as pd

# Save overall metrics
overall_metrics_df = pd.DataFrame.from_dict(
    {
        model: {
            metric: values[metric]
            for metric in ["Accuracy", "F1 Score", "Precision", "Recall"]
        }
        for model, values in results.items()
    },
    orient="index",
)
overall_metrics_df.to_excel("overall_metrics.xlsx")

# Save per-class metrics
with pd.ExcelWriter("per_class_metrics.xlsx") as writer:
    for model, values in results.items():
        class_report_df = pd.DataFrame(values["Class Report"]).T
        class_report_df.to_excel(writer, sheet_name=model)

# %%
overall_metrics_df = pd.read_excel("overall_metrics.xlsx", index_col=0)
per_class_metrics_df = pd.read_excel("per_class_metrics.xlsx", sheet_name=None)

# %%
