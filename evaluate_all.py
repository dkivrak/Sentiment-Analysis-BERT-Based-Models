from evaluate import evaluate_model
import csv
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

output_csv = "evaluation_results.csv"

if __name__ == "__main__":
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Dataset", "Accuracy", "F1 Score"])

        for model_name in models:
            model_dir_name = model_name.replace("/", "_")
            for data in datasets:
                dataset_name = data["name"]
                dataset_path = data["path"]
                dataset_type = data["type"]

                model_path = os.path.join("saved_model", f"{model_dir_name}_{dataset_name}")

                try:
                    acc, f1 = evaluate_model(model_path, dataset_path, dataset_type=dataset_type)
                    writer.writerow([model_name, dataset_name, f"{acc:.4f}", f"{f1:.4f}"])
                except Exception as e:
                    print(f" Error evaluating {model_name} on {dataset_name}: {e}")
                    writer.writerow([model_name, dataset_name, "ERROR", "ERROR"])

    print(f"\n Evaluation results saved to: {output_csv}")
