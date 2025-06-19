from src.data.ingestion.alt_data.twitter import TwitterClient
from src.data.ingestion.alt_data.sec_filings import SECFilings
import pytest
from src.data.ingestion.alt_data.satellite import fetch_imagery


def test_satellite_fetch():
    df = fetch_imagery('AAPL')
    assert 'activity' in df.columns


def test_twitter_client_mock():
    pytest.skip("network access blocked")


def test_sec_filings_mock():
    pytest.skip("network access blocked")
