# Deep Learning Computer Vision Project

## Overview

This project involves using deep learning techniques for computer vision tasks. The project includes multiple models for image classification, leveraging architectures like CNN, ResNet-101, and Vision Transformer (ViT). Additionally, a Streamlit app is provided for easy model interaction.

## Installation

### Prerequisites

To get started, you need to install the required dependencies. 

```bash
pip install -r requirements.txt
```

## Streamlit App

To run the Streamlit app:

1. Navigate to the application directory.

```bash
cd Code/app
```

2. Download the pre-trained models.

```bash
wget https://storage.googleapis.com/dl-grp-4-bucket/model/model_cnn.pt
wget https://storage.googleapis.com/dl-grp-4-bucket/model/model_resnet_101.pt
wget https://storage.googleapis.com/dl-grp-4-bucket/model/model_vit_b_16.pt
```

3. Start the Streamlit server.

```bash
python3 -m streamlit run image_classifier_app.py --server.port=8888
```

## Dataset

To download the dataset, you need a Kaggle account. 

1. Create an account on Kaggle and generate an API token under your profile settings. Save the token as `~/.kaggle/kaggle.json`.

```bash
vim ~/.kaggle/kaggle.json
```

2. Accept the competition rules on the Kaggle competition page:  
   [Kaggle Competition Page](https://www.kaggle.com/competitions/imaterialist-fashion-2019-FGVC6/overview).

3. Download the dataset using the provided script:

```bash
chmod +x download_data/download_data.sh
./download_data/download_data.sh
```

4. Process the dataset by running all cells in the following Jupyter Notebook:

```bash
download_data/final_dataset_excel_creation.ipynb
```

### Alternative Dataset Download

To download the dataset directly:

```bash
wget https://storage.googleapis.com/dl-grp-4-bucket/dataset.zip
unzip dataset.zip
```

## Training Models

The provided scripts allow you to train different models. Ensure the `PATH` variable in each script points to the parent directory where the dataset folder is located.

### CNN Model

```bash
python3 Code/train_cnn.py
```

### ResNet-101 Model

```bash
python3 Code/train_resnet_50.py
```

### ViT Model

```bash
python3 Code/Code/train_torch_vit.py
```
