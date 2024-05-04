import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
overall_metrics_df = pd.read_excel("overall_metrics.xlsx", index_col=0)
per_class_metrics_df = pd.read_excel("per_class_metrics.xlsx", sheet_name=None)

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


@st.cache_resource
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


def preprocess_image(image, model_name):
    if model_name == "ViT_B_16":
        transform = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
    elif model_name == "ResNet_101":
        transform = models.ResNet101_Weights.DEFAULT.transforms()
    elif model_name == "CNN":
        transform = cnn_transform
    return transform(image).unsqueeze(0).to(device)


st.set_page_config(
    page_title="Fashion Finder",
    page_icon="🕴️",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Fashion Finder: Classify Clothing Categories with AI")
st.write(
    "Upload an image and let our AI models identify the fashion categories present in it."
)

st.sidebar.title("Options")
model_name = st.sidebar.selectbox("Select Model", ["ViT_B_16", "ResNet_101", "CNN"])
threshold = st.sidebar.slider(
    "Probability Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05
)

model_path = f"model_{model_name.lower()}.pt"
model = load_model(model_name, model_path)
st.subheader("Overall Metrics")
with st.expander("Expand to view overall metrics"):
    overall_metrics = overall_metrics_df.loc[model_name]
    st.write(overall_metrics)

st.subheader("Per-Class Metrics")
with st.expander("Expand to view per-class metrics"):
    per_class_metrics = per_class_metrics_df[model_name]
    st.write(per_class_metrics)

st.subheader(f"Selected Model: {model_name}")
# st.write("Model Description: ...")

uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        input_tensor = preprocess_image(image, model_name)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.sigmoid(output).squeeze().tolist()

        st.subheader("Classification Results")
        results = []
        for i, probability in enumerate(probabilities):
            if probability > threshold:
                results.append((category_mapping[i], probability))

        st.table(pd.DataFrame(results, columns=["Apparel", "Score"]))

st.markdown("---")
st.subheader("About Fashion Finder")
st.write(
    "Fashion Finder is an AI-powered app that helps you identify fashion categories in images. It utilizes state-of-the-art deep learning models to classify clothing items and accessories with high accuracy."
)
st.write("Key Features:")
st.write(
    "- Support for multiple deep learning models ViT, ResNet, CNN and many more coming..."
)
st.write("- Customizable probability threshold for fine-grained control")
st.write("Explore the power of AI in fashion analysis with Fashion Finder!")
