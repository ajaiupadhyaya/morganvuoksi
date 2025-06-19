"""SEC filings downloader using sec-api or mock data."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

try:
    from sec_api import QueryApi  # type: ignore
except Exception:  # pragma: no cover - optional dep
    QueryApi = None


@dataclass
class Filing:
    cik: str
    form_type: str
    filing_date: datetime
    report_url: str


class SECFilings:
    """Retrieve SEC filings and return parsed data."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api = QueryApi(api_key) if QueryApi and api_key else None

    def fetch_filings(self, ticker: str, form_types: List[str] | None = None) -> pd.DataFrame:
        form_types = form_types or ["10-K", "10-Q"]
        if self.api:
            try:
                query = {
                    "query": {
                        "query_string": {
                            "query": f"ticker:{ticker} AND formType:({' OR '.join(form_types)})"
                        }
                    },
                    "from": "0",
                    "size": "10",
                    "sort": [{"filedAt": {"order": "desc"}}]
                }
                data = self.api.get_filings(query)
                filings = [
                    {
                        "cik": d.get("cik"),
                        "form_type": d.get("formType"),
                        "filing_date": d.get("filedAt"),
                        "report_url": d.get("linkToFilingDetails"),
                    }
                    for d in data.get("filings", [])
                ]
            except Exception:  # pragma: no cover - network issues
                filings = []
        else:
            filings = [
                {
                    "cik": "0000320193",
                    "form_type": ft,
                    "filing_date": datetime.utcnow().isoformat(),
                    "report_url": "https://example.com",
                }
                for ft in form_types
            ]

        df = pd.DataFrame(filings)
        if not df.empty:
            df["filing_date"] = pd.to_datetime(df["filing_date"])
        return df
