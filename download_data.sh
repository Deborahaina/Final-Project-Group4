#!/bin/bash
#!/usr/bin/env python3

# Install Kaggle API
pip3 install kaggle

# Set the competition name and dataset name
competition_name="imaterialist-fashion-2019-FGVC6"
dataset_name="imaterialist-fashion-2019-FGVC6.zip"

# Create a directory to store the dataset
mkdir -p dataset

# Download the dataset
~/.local/bin/kaggle competitions download -c $competition_name -p dataset

# Extract the dataset
unzip -q dataset/$dataset_name -d dataset

# Remove the zip file
rm dataset/$dataset_name

echo "Dataset downloaded and extracted successfully!"