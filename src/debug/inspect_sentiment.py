import pandas as pd

df = pd.read_csv("data/processed/sentiment_scores.csv")
print("🔥 HEAD:")
print(df.head(10))
print("\n🧠 COLUMNS:", df.columns.tolist())
