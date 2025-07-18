from train import train_model
import os

models = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    "xlm-roberta-base",
    "albert-base-v2"
]

datasets = [
    {"name": "imdb", "path": "data/imdb_train.csv", "type": "generic"},
    {"name": "yelp", "path": "data/yelp_train.csv", "type": "generic"},
    {"name": "amazon", "path": "data/amazon_train.csv", "type": "amazon"}
]

base_output_dir = "saved_model"

if __name__ == "__main__":
    for model_name in models:
        for data in datasets:
            dataset_name = data["name"]
            dataset_path = data["path"]
            dataset_type = data["type"]

            output_dir = os.path.join(base_output_dir, f"{model_name.replace('/', '_')}_{dataset_name}")
            train_model(
                csv_path=dataset_path,
                model_name=model_name,
                output_dir=output_dir,
                dataset_type=dataset_type
            )
