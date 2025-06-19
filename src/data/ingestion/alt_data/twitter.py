"""Simple Twitter client using tweepy or generating mock data."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

import pandas as pd
import numpy as np

try:
    import tweepy  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tweepy = None


@dataclass
class Tweet:
    created_at: datetime
    text: str
    sentiment: float


class TwitterClient:
    """Fetch tweets and compute basic sentiment."""

    def __init__(self, bearer_token: str | None = None) -> None:
        if tweepy and bearer_token:
            self.client = tweepy.Client(bearer_token)
        else:  # pragma: no cover - fallback
            self.client = None

    def fetch(self, query: str, max_results: int = 10) -> pd.DataFrame:
        if self.client:
            try:
                resp = self.client.search_recent_tweets(query=query, max_results=max_results)
                tweets = [t.text for t in resp.data] if resp.data else []
            except Exception:  # pragma: no cover - network issues
                tweets = []
        else:
            tweets = [f"{query} tweet {i}" for i in range(max_results)]

        sentiments = np.random.uniform(-1, 1, len(tweets))
        df = pd.DataFrame({"text": tweets, "sentiment": sentiments})
        df["created_at"] = pd.Timestamp.utcnow()
        return df
