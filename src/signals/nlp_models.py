# src/signals/nlp_models.py
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import torch

tqdm.pandas()

def load_headlines(file_path):
    df = pd.read_csv(file_path)
    if not all(col in df.columns for col in ["date", "symbol", "headline"]):
        raise ValueError("CSV must have columns: date, symbol, headline")
    return df

def run_sentiment_analysis(df, text_column="headline"):
    device = 0 if torch.cuda.is_available() else -1 if not torch.backends.mps.is_available() else "mps"
    print(f"Device set to use {device}")
    nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device, framework="pt")

    df = df.dropna(subset=[text_column])
    df[text_column] = df[text_column].astype(str)

    print("Running sentiment analysis...")
    sentiments = df[text_column].progress_apply(lambda x: nlp(x)[0])
    
    df["sentiment_label"] = sentiments.apply(lambda x: x["label"])
    df["sentiment_score"] = sentiments.apply(lambda x: x["score"])
    return df

def main():
    raw_headlines = load_headlines("data/raw/news_headlines.csv")
    scored_df = run_sentiment_analysis(raw_headlines)
    scored_df.to_csv("data/processed/sentiment_scores.csv", index=False)
    print("[✓] Sentiment analysis complete → data/processed/sentiment_scores.csv")

if __name__ == "__main__":
    main()