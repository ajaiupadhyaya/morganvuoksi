# src/signals/nlp_models.py
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import torch
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")  

tqdm.pandas()

def fetch_news(symbols, from_days_ago=1):
    all_articles = []
    for symbol in symbols:
        url = f"https://newsapi.org/v2/everything"
        params = {
            "q": symbol,
            "language": "en",
            "sortBy": "publishedAt",
            "from": (datetime.now() - timedelta(days=from_days_ago)).strftime("%Y-%m-%d"),
            "apiKey": NEWSAPI_KEY,
            "pageSize": 50
        }
        response = requests.get(url, params=params)
        data = response.json()
        for article in data.get("articles", []):
            all_articles.append({
                "date": article["publishedAt"][:10],
                "symbol": symbol,
                "headline": article["title"]
            })
    return pd.DataFrame(all_articles)

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
    symbols = ["AAPL", "MSFT", "NVDA", "GOOGL"]  # ✅ Eventually, load dynamically from your portfolio universe
    raw_headlines = fetch_news(symbols)
    scored_df = run_sentiment_analysis(raw_headlines)
    scored_df.to_csv("data/processed/sentiment_scores.csv", index=False)
    print("[✓] Sentiment analysis complete → data/processed/sentiment_scores.csv")

if __name__ == "__main__":
    main()
