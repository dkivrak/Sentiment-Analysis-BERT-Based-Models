# Sentiment-Analysis-BERT-Based-Models
A project comparing 5 different BERT-based models for sentiment analysis.

## Purpose
The objective is to evaluate transformer-based language models’ effectiveness in sentiment analysis and to compare their results.


## Models Used
- `bert-base-uncased`
- `distilbert-base-uncased`
- `roberta-base`
- `xlm-roberta-base`
- `albert-base-v2`

## Dataset Used
- IMDb (`data/imdb_train.csv`) — general movie reviews                                            Dataset Link: https://huggingface.co/datasets/stanfordnlp/imdb

## Side Notes
Due to hardware limitations, the original datasets have been reduced in size, and hyperparameters have been tuned conservatively.
You can contact me for any questions or feedback.


## Project Structure

data/               # Dataset CSV files
├── imdb_train.csv


preprocessing.py    # Data loading and tokenization utilities  
model.py            # Model definition class  
train.py            # Training script  
run_experiment.py   # Script to run all experiments in loops  
evaluate.py         # Model evaluation script  
evaluate_all.py     # Evaluate multiple models  
requirements.txt    # Project dependencies  
README.md           # Project overview (this file)
