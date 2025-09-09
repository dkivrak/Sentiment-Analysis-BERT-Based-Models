# Sentiment-Analysis-BERT-Based-Models
A project comparing 5 different BERT-based models for sentiment analysis.

## Purpose
The objective of this project is to evaluate transformer-based language models’ effectiveness in sentiment analysis and to compare their results under a unified experimental setup.

Sentiment analysis is crucial for understanding user opinions in domains such as product reviews, movie feedback, and service evaluations. By comparing different models, this project highlights the trade-offs between accuracy, F1-score, and computational efficiency.

## Models Used
Five transformer-based models were fine-tuned using the HuggingFace transformers library:
- `bert-base-uncased`
- `distilbert-base-uncased`
- `roberta-base`
- `xlm-roberta-base`
- `albert-base-v2`

## Dataset Used
- IMDb (`data/imdb_train.csv`) — general movie reviews                                            Dataset Link: https://huggingface.co/datasets/stanfordnlp/imdb
-Yelp Reviews (Restaurant/business reviews)
-Amazon Reviews (Product reviews, preprocessed by concatenating title + content)

## Side Notes
Due to hardware limitations, the original datasets have been reduced in size, and hyperparameters have been tuned conservatively.
You can contact me for any questions or feedback.

## Methodology
1) Preprocessing 
-    Tokenization and padding using HuggingFace tokenizerTruncation at 256 tokens
-    Label normalization (positive/negative)
-    For Amazon, combined title + content for single input field

2) Training Setup
-    Optimizer:
-    AdamW (default HuggingFace settings)
-    Batch size: 16
-    Epochs: 1 (limited by hardware)
-    Trainer API used for model fine-tuningBalanced class distributions
   
 3) Evaluation Metrics
-    Accuracy
-    F1-score (captures precision + recall balance)

## Project Structure
├── data/
│   └── imdb_train.csv   # Dataset CSV files
├── preprocessing.py      # Data loading & tokenization
├── model.py              # Model definitions
├── train.py / run_experiment.py
├── evaluate.py           # Single model evaluation
├── evaluate_all.py       # Compare all models
├── requirements.txt      # Dependencies
└── README.md             # Project documentation

## Discussion & Insights
-RoBERTa and ALBERT achieved the best overall performance, especially on Amazon reviews.
-DistilBERT gave competitive results with far fewer resources, making it practical for real-world applications.
-XLM-RoBERTa struggled on English-only datasets (IMDB, Amazon), likely due to its multilingual nature.
-Dataset characteristics strongly influenced outcomes:
-Amazon dataset was easiest (clearer sentiment, well-balanced).
-IMDB was hardest (long, nuanced reviews).

## Installation
# Clone repo
git clone https://github.com/dkivrak/Sentiment-Analysis-BERT-Based-Models.git
cd Sentiment-Analysis-BERT-Based-Models

# Install dependencies
pip install -r requirements.txt

# Run training
python run_experiment.py

# Evaluate results
python evaluate.py --model bert-base-uncased
