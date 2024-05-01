import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models

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

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image, model_name):
    if model_name == "ViT_B_16":
        transform = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
    elif model_name == "ResNet_101":
        transform = models.ResNet101_Weights.DEFAULT.transforms()
    return transform(image).unsqueeze(0).to(device)


st.set_page_config(
    page_title="Fashion/Apparel Multi Label Image Classifier",
    page_icon="🕴️",
    layout="centered",
)

st.title("🕴️ Fashion/Apparel Multi Label Image Classifier")

st.sidebar.title("Options")
threshold = st.sidebar.slider(
    "Probability Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05
)
model_name = st.sidebar.selectbox("Select Model", ["ViT_B_16", "ResNet_101"])

# Load the selected model
model_path = f"model_{model_name.lower()}.pt"
model = load_model(model_name, model_path)

uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if model and uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = preprocess_image(image, model_name)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.sigmoid(output).squeeze().tolist()

    st.subheader("Classification Results")
    for i, probability in enumerate(probabilities):
        if probability > threshold:
            st.write(f"{category_mapping[i]}: {probability:.2f}")
