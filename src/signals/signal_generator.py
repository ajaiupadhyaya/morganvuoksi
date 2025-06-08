import pandas as pd
from tqdm import tqdm
from src.utils.helpers import ensure_dir

# Set input paths
ALPHA_PATH = "data/processed/alpha_factors.parquet"
SENTIMENT_PATH = "data/processed/sentiment_scores.csv"
OUTPUT_PATH = "data/processed/trading_signals.csv"

def load_data(alpha_path: str, sentiment_path: str):
    alpha_df = pd.read_parquet(alpha_path)
    sentiment_df = pd.read_csv(sentiment_path, parse_dates=["date"])
    return alpha_df, sentiment_df

def merge_data(alpha_df, sentiment_df):
    # Ensure date columns are in datetime format
    alpha_df.index = pd.to_datetime(alpha_df.index)
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

    merged = pd.merge(
        sentiment_df,
        alpha_df,
        left_on='date',
        right_index=True,
        how='inner'
    )
    return merged

def generate_signals(df):
    def decide(row):
        if (
            row["momentum_10d"] > 0 and
            row["rsi"] < 30 and
            row["sentiment_label"] == "positive"
        ):
            return "buy"
        elif (
            row["momentum_10d"] < 0 and
            row["rsi"] > 70 and
            row["sentiment_label"] == "negative"
        ):
            return "sell"
        else:
            return "hold"
    df["signal"] = df.apply(decide, axis=1)
    return df

def main():
    print("🚀 Generating trading signals...")
    ensure_dir("data/processed")

    # Load
    alpha_df, sentiment_df = load_data(ALPHA_PATH, SENTIMENT_PATH)

    # Merge
    combined_df = merge_data(alpha_df, sentiment_df)

    # Generate signals
    signals_df = generate_signals(combined_df)

    # Save
    signals_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[✓] Trading signals saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()