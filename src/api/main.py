from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

app = FastAPI(
    title="MorganVuoksi API",
    description="Powering the next-generation financial terminal.",
    version="0.1.0"
)

# --- CORS Middleware ---
# This allows our Next.js frontend (running on a different port) to communicate with this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MOCK DATA GENERATION ---
# This logic is moved from the Streamlit app to the backend.

def get_mock_data():
    """Returns mock data for the terminal."""
    price_data = pd.DataFrame({
        'time': pd.to_datetime(pd.date_range('2023-01-01', periods=90, freq='D')),
        'price': np.random.randn(90).cumsum() + 130
    })

    return {
        "symbol": {
            "name": "Alphabet Inc.",
            "ticker": "GOOGL",
            "price": 134.33,
            "change_val": 1.12,
            "change_pct": 0.84,
            "volume": "24.5M",
            "market_cap": "1.67T"
        },
        "price_chart": {
            "1D": price_data.tail(1).to_dict('records'),
            "5D": price_data.tail(5).to_dict('records'),
            "1M": price_data.tail(30).to_dict('records'),
            "1Y": price_data.to_dict('records'),
        },
        "watchlist": [
            {"ticker": "AAPL", "price": 172.25, "change_pct": 1.24},
            {"ticker": "GOOGL", "price": 134.33, "change_pct": -0.54},
            {"ticker": "BTC-USD", "price": 42050.78, "change_pct": 2.11},
            {"ticker": "TSLA", "price": 240.15, "change_pct": -1.03},
        ],
        "headlines": [
            {"source": "Reuters", "title": "Fed hints at rate pause — markets surge on renewed optimism."},
            {"source": "Bloomberg", "title": "AI chip demand continues to lift the entire semiconductor sector."},
            {"source": "CoinDesk", "title": "Crypto ETF chatter reignites bull run as institutional interest grows."},
        ],
        "key_executives": [
            {"name": "Sundar Pichai", "title": "CEO / Director"},
            {"name": "Ruth Porat", "title": "President / CIO / CFO"},
            {"name": "Prabhakar Raghavan", "title": "SVP, Google Search"},
        ]
    }

# --- API ENDPOINTS ---

@app.get("/")
def read_root():
    return {"status": "MorganVuoksi API is running."}

@app.get("/api/v1/terminal_data")
def get_terminal_data():
    """
    The primary endpoint to fetch all the data needed for the main terminal view.
    """
    return get_mock_data()

@app.get("/api/v1/dcf/{symbol}")
def get_dcf_valuation(symbol: str):
    """
    A mock endpoint for DCF valuation.
    """
    return {
        "symbol": symbol,
        "estimated_value": np.random.randint(150, 250),
        "growth_rate": f"{np.random.uniform(0.04, 0.06):.2%}",
        "margin_of_safety": "20%"
    } 