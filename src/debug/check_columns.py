# src/debug/check_columns.py

import pandas as pd

alpha_path = "data/processed/alpha_factors.parquet"
sentiment_path = "data/processed/sentiment_scores.csv"

alpha_df = pd.read_parquet(alpha_path)
sentiment_df = pd.read_csv(sentiment_path)

print("🔎 Alpha Columns:", alpha_df.columns.tolist())
print("🔎 Sentiment Columns:", sentiment_df.columns.tolist())
print("✅ Alpha Sample:")
print(alpha_df.head())
print("✅ Sentiment Sample:")
print(sentiment_df.head())
