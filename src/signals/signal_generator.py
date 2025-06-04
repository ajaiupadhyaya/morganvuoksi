import pandas as pd
import numpy as np
from pathlib import Path

def load_data(alpha_path: str, sentiment_path: str):
    alpha_df = pd.read_parquet(alpha_path)
    sentiment_df = pd.read_parquet(sentiment_path)
    return alpha_df, sentiment_df

def merge_data(alpha_df, sentiment_df):
    df = pd.merge(alpha_df, sentiment_df, on=["date", "ticker"], how="inner")
    df = df.dropna()
    return df

def normalize_scores(df, alpha_cols, sentiment_col):
    # Z-score normalization
    for col in alpha_cols:
        df[col + "_z"] = df.groupby("date")[col].transform(lambda x: (x - x.mean()) / x.std())

    df["sentiment_z"] = df.groupby("date")[sentiment_col].transform(lambda x: (x - x.mean()) / x.std())
    return df

def compute_signal(df, alpha_cols, sentiment_col, alpha_weight=0.7, sentiment_weight=0.3):
    # Weighted composite score
    z_cols = [col + "_z" for col in alpha_cols]
    df["alpha_score"] = df[z_cols].mean(axis=1)
    df["composite_score"] = alpha_weight * df["alpha_score"] + sentiment_weight * df["sentiment_z"]

    # Rank signals per day
    df["rank"] = df.groupby("date")["composite_score"].rank(method="first", ascending=False)
    df["signal"] = df.groupby("date")["composite_score"].transform(
        lambda x: pd.qcut(x, q=3, labels=[-1, 0, 1]).astype(int)
    )
    return df

def save_signals(df, output_path):
    df_out = df[["date", "ticker", "composite_score", "signal"]].copy()
    df_out.to_parquet(output_path, index=False)
    print(f"[✓] Signals saved to: {output_path}")

def main():
    alpha_path = "data/processed/alpha_factors.parquet"
    sentiment_path = "data/processed/sentiment_scores.parquet"
    output_path = "data/processed/signals.parquet"

    alpha_df, sentiment_df = load_data(alpha_path, sentiment_path)

    df = merge_data(alpha_df, sentiment_df)

    alpha_columns = [col for col in alpha_df.columns if col not in ["date", "ticker"]]
    sentiment_column = "sentiment_score"

    df = normalize_scores(df, alpha_columns, sentiment_column)
    df = compute_signal(df, alpha_columns, sentiment_column)
    save_signals(df, output_path)

if __name__ == "__main__":
    main()