import streamlit as st
import yfinance as yf
import pandas as pd
import joblib

from src.features import add_technical_indicators
from src.logger import logger

# ==============================
# Load Model Artifacts
# ==============================
model = joblib.load("models/xgb_model.pkl")
feature_columns = joblib.load("models/feature_columns2.pkl")
reverse_label_map = joblib.load("models/label_map.pkl")

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="ML Trading Signal", layout="wide")

st.title("📈 ML-Based Buy / Hold / Sell Prediction System")

symbol = st.text_input("Enter Stock Symbol (e.g., ^BSESN, AAPL, RELIANCE.NS)", "^BSESN")

if st.button("Generate Signal"):

    # ==============================
    # Fetch Data
    # ==============================
    st.write("Fetching latest data...")
    df = yf.download(symbol, period="5y", auto_adjust=False)

    # 🔥 Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        st.error("Invalid symbol or no data found.")
        st.stop()

    # ==============================
    # Feature Engineering
    # ==============================
    logger.info("Adding advanced technical indicators...")
    df = add_technical_indicators(df)

    # Make sure required features exist
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        st.error(f"Missing features: {missing}")
        st.stop()

    # ==============================
    # Prepare Live Input
    # ==============================
    X_live = df[feature_columns].tail(1)

    # ==============================
    # Prediction
    # ==============================
    raw_pred = model.predict(X_live)[0]
    prediction = reverse_label_map[raw_pred]

    proba = model.predict_proba(X_live)[0]
    confidence = round(max(proba) * 100, 2)

    # ==============================
    # Display Signal
    # ==============================
    if prediction == 1:
        st.success(f"🟢 BUY NOW ({confidence}% confidence)")
    elif prediction == -1:
        st.error(f"🔴 SELL ({confidence}% confidence)")
    else:
        st.warning(f"🟡 HOLD ({confidence}% confidence)")

    # ==============================
    # Show Chart
    # ==============================
    st.subheader("Price Chart")
    st.line_chart(df["Close"])

    st.subheader("Recent Data")
    st.dataframe(df.tail())