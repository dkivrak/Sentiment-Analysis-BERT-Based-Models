import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from preprocessing import load_csv_dataset, load_amazon_dataset, tokenize_dataset
import torch
import os

def evaluate_model(model_path, test_csv, dataset_type="generic"):
    print(f"\n Evaluating model at {model_path} on {test_csv} \n")

    if dataset_type == "amazon":
        dataset = load_amazon_dataset(test_csv)
    else:
        dataset = load_csv_dataset(test_csv)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    tokenized = tokenize_dataset(dataset, tokenizer)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    model.eval()

    all_preds = []
    all_labels = []

    for batch in tokenized:
        input_ids = batch["input_ids"].unsqueeze(0)
        attention_mask = batch["attention_mask"].unsqueeze(0)
        label = batch["label"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()

        all_preds.append(pred)
        all_labels.append(label)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f" Accuracy: {acc:.4f}")
    print(f" F1 Score: {f1:.4f}")

    return acc, f1
