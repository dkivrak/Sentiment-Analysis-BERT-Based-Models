import pandas as pd
from datasets import Dataset

def load_csv_dataset(path, text_col="text", label_col="label"):
    df = pd.read_csv(path)
    df = df[[text_col, label_col]].dropna()
    df.rename(columns={text_col: "text", label_col: "label"}, inplace=True)
    return Dataset.from_pandas(df)


def load_amazon_dataset(path):
    df = pd.read_csv(path)
    df["text"] = df["title"].fillna("") + " " + df["content"].fillna("")
    df = df[["text", "label"]].dropna()
    return Dataset.from_pandas(df)


def tokenize_dataset(dataset, tokenizer, max_length=256):
    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    return dataset.map(tokenize_fn, batched=True)
