# %%
from datasets import DatasetDict, Dataset
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import torch

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
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

# %%
ds = DatasetDict({"train": train_dataset, "test": test_dataset})
# %%
from transformers import ViTFeatureExtractor

# %%
model_name_or_path = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
# %%
feature_extractor


# %%
def process_example(example):
    file = DATA_DIR + example["ImageId"]
    y = example["target_class"]
    y = y.split(",")
    labels_ohe = [int(e) for e in y]
    y = torch.FloatTensor(labels_ohe)
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs = feature_extractor(img, return_tensors="pt")
    inputs["labels"] = example["target_class"]
    return inputs


# %%
process_example(ds["train"][0])

# %%


def transform(example_batch):
    # Initialize lists to store the processed images and labels
    input_ids = []
    labels = []
    # Process each example in the batch
    for i in range(len(example_batch["ImageId"])):
        example = {
            "ImageId": example_batch["ImageId"][i],
            "target_class": example_batch["target_class"][i],
        }

        # Process the example using the process_example function
        processed_example = process_example(example)

        # Extract the processed inputs and labels
        input_ids.append(processed_example["pixel_values"])

        # Convert the labels to a list of integers
        label_list = processed_example["labels"].split(",")
        label_list = [int(label) for label in label_list]
        labels.append(label_list)

    # Convert lists to tensors
    input_ids = torch.cat(input_ids, dim=0)
    labels = torch.tensor(labels, dtype=torch.float32)
    return {
        "pixel_values": input_ids,
        "labels": labels,
    }


# %%
prepared_ds = ds.with_transform(transform)
# %%
prepared_ds["train"][0:2]

# %%

import torch


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


# %%
import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

# %%


def compute_metrics(p):
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
    )


# %%


from transformers import ViTForImageClassification

# %%

labels = ds["train"].features["target_class"]

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=46,
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
    problem_type="multi_label_classification",
)
# %%


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./vit-base-v1",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=4,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="tensorboard",
    load_best_model_at_end=True,
)


from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["validation"],
    tokenizer=feature_extractor,
)

train_results = trainer.train()

# %%
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(prepared_ds["validation"])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
