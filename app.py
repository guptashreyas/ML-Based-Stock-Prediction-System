import streamlit as st
import yfinance as yf
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.features import add_technical_indicators
from src.config import MODEL_PATH

st.set_page_config(page_title="ML Trading Signal", layout="wide")

st.title("📈 ML-Based Buy / Hold / Sell Predictor")

# ========================
# User Input
# ========================

symbol = st.text_input("Enter Stock Symbol (Example: AAPL, ^BSESN, ^NSEI)", "^BSESN")

if st.button("Generate Signal"):

    # Download data
    df = yf.download(symbol, period="2y",auto_adjust=True)

    if df.empty:
        st.error("Invalid symbol or no data found.")
        st.stop()

    # Fix MultiIndex if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)

    # Add features
    df = add_technical_indicators(df)

    # Load trained model
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load("models/feature_columns.pkl")

    # Prepare last row for prediction
    latest = df.iloc[-1:]
    X_live = latest[feature_columns]

    prediction = model.predict(X_live)[0]
    probabilities = model.predict_proba(X_live)[0]

    signal_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}
    signal = signal_map[prediction]
    confidence = round(max(probabilities) * 100, 2)

    # ========================
    # Signal Display
    # ========================

    st.subheader("📊 Current Signal")

    if signal == "BUY":
        st.success(f"✅ BUY NOW (Confidence: {confidence}%)")
    elif signal == "SELL":
        st.error(f"🔴 SELL (Confidence: {confidence}%)")
    else:
        st.warning(f"🟡 HOLD (Confidence: {confidence}%)")

    # ========================
    # Plot Chart
    # ========================

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["EMA_20"],
        mode="lines",
        name="EMA 20"
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["EMA_50"],
        mode="lines",
        name="EMA 50"
    ))

    st.plotly_chart(fig, use_container_width=True)