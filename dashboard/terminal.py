import asyncio
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.fetcher import DataFetcher
from src.signals.ml_models import ModelLoader
from src.backtesting.engine import BacktestEngine
from src.portfolio.optimizer import mean_variance_optimizer, expected_returns, covariance_matrix
from src.execution.options_engine import OptionsEngine
from fundamentals.dcf import discounted_cash_flow


# ----- Helper Functions -----
async def _fetch_data(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    fetcher = DataFetcher()
    return await fetcher.fetch_stock_data(symbol, start, end)


def fetch_data(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    return asyncio.run(_fetch_data(symbol, start, end))


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["Returns"] = data["Close"].pct_change()
    data["Volatility"] = data["Returns"].rolling(20).std()
    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))
    return data.dropna()


# ----- Streamlit Layout -----
st.set_page_config(page_title="MorganVuoksi Terminal", layout="wide")

st.title("MorganVuoksi Quant Terminal")

if "market_data" not in st.session_state:
    st.session_state["market_data"] = None

if "dcf_cache" not in st.session_state:
    st.session_state["dcf_cache"] = {}

# Sidebar Inputs
symbol = st.sidebar.text_input("Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
model_choice = st.sidebar.selectbox("Model", ["xgboost", "lstm", "transformer"])

if st.sidebar.button("Load Market Data"):
    with st.spinner("Fetching market data..."):
        st.session_state["market_data"] = fetch_data(symbol, pd.to_datetime(start_date), pd.to_datetime(end_date))
        st.success("Data loaded")


tabs = st.tabs([
    "📈 Market Data Viewer",
    "🧠 ML Model Predictions",
    "⚙️ Backtesting",
    "📊 Portfolio Optimizer",
    "📝 Trade History Viewer",
    "💸 DCF Valuation Module",
])

# ----- Market Data Viewer -----
with tabs[0]:
    st.header("Market Data Viewer")
    data = st.session_state.get("market_data")
    if data is not None:
        st.line_chart(data["Close"])
        st.dataframe(data.tail())
        csv = data.to_csv().encode("utf-8")
        st.download_button("Download CSV", csv, file_name=f"{symbol}_data.csv")
    else:
        st.info("Load data from the sidebar to view market information.")

# ----- ML Model Predictions -----
with tabs[1]:
    st.header("ML Model Predictions")
    data = st.session_state.get("market_data")
    if data is None:
        st.info("Load market data first.")
    else:
        feat = prepare_features(data)
        train_size = int(len(feat) * 0.8)
        X_train = feat.iloc[:train_size][["Returns", "Volatility", "MA20", "MA50", "RSI"]]
        y_train = feat["Close"].shift(-1).dropna().iloc[:train_size]
        X_test = feat.iloc[train_size:][["Returns", "Volatility", "MA20", "MA50", "RSI"]]
        y_test = feat["Close"].shift(-1).dropna().iloc[train_size:]

        if st.button("Train & Predict"):
            loader = ModelLoader()
            if model_choice == "xgboost":
                model = loader.load_xgboost()
            elif model_choice == "lstm":
                model = loader.load_lstm()
            else:
                model = loader.load_transformer()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            res = pd.DataFrame({"Actual": y_test.values, "Predicted": preds}, index=y_test.index)
            fig = px.line(res, y=["Actual", "Predicted"], title="Model Prediction")
            st.plotly_chart(fig, use_container_width=True)
            rmse = ((res["Actual"] - res["Predicted"]) ** 2).mean() ** 0.5
            st.metric("RMSE", f"{rmse:.4f}")

# ----- Backtesting -----
with tabs[2]:
    st.header("Backtesting")
    data = st.session_state.get("market_data")
    if data is None:
        st.info("Load market data first.")
    else:
        if st.button("Run Backtest"):
            df = data.copy()
            ma = df["Close"].rolling(20).mean()
            signals = (df["Close"] > ma).astype(int) - (df["Close"] < ma).astype(int)
            engine = BacktestEngine({})
            results = engine.run_backtest(df.dropna(), signals.fillna(0))
            st.line_chart(results["portfolio"]["total"])
            st.write(results["metrics"])
            csv = results["portfolio"].to_csv().encode("utf-8")
            st.download_button("Download Portfolio", csv, file_name="backtest.csv")

# ----- Portfolio Optimizer -----
with tabs[3]:
    st.header("Portfolio Optimizer")
    tickers = st.text_input("Tickers (comma separated)", "AAPL,MSFT,GOOG")
    if st.button("Optimize Portfolio"):
        syms = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        price_data = {s: fetch_data(s, pd.to_datetime(start_date), pd.to_datetime(end_date)) for s in syms}
        prices = pd.DataFrame({k: v["Close"] for k, v in price_data.items()})
        returns = prices.pct_change().dropna()
        mu = expected_returns(returns)
        cov = covariance_matrix(returns)
        weights = mean_variance_optimizer(mu, cov)
        st.bar_chart(weights)
        st.dataframe(weights.rename("weight"))

# ----- Trade History Viewer -----
with tabs[4]:
    st.header("Trade History")
    engine = OptionsEngine(paper=True)
    history = engine.get_history()
    if history.empty:
        st.info("No trades recorded.")
    else:
        st.dataframe(history)
        csv = history.to_csv().encode("utf-8")
        st.download_button("Download History", csv, file_name="trades.csv")

# ----- DCF Valuation Module -----
with tabs[5]:
    st.header("DCF Valuation")
    wacc = st.number_input("WACC", value=0.1)
    growth = st.number_input("Terminal Growth", value=0.02)
    if st.button("Run DCF"):
        result = discounted_cash_flow(symbol, wacc=wacc, terminal_growth=growth)
        st.session_state["dcf_cache"][symbol] = result.present_value
        st.success(f"Present Value: ${result.present_value:,.2f}")
        st.json(result.assumptions)
