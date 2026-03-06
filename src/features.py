import numpy as np
import pandas as pd
from src.logger import logger


def add_technical_indicators(df):

    logger.info("Adding advanced technical indicators...")

    # =====================================
    # FIX 1: Flatten MultiIndex Columns
    # =====================================
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()

    # Ensure required columns exist
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # =====================================
    # EMA
    # =====================================
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    df["EMA_Spread"] = df["EMA_20"] - df["EMA_50"]

    # Avoid division by zero
    df["Dist_from_EMA20"] = np.where(
        df["EMA_20"] != 0,
        (df["Close"] - df["EMA_20"]) / df["EMA_20"],
        0
    )

    # =====================================
    # RSI
    # =====================================
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # =====================================
    # MACD
    # =====================================
    ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # =====================================
    # Bollinger Bands
    # =====================================
    sma = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()

    df["BB_Upper"] = sma + 2 * std
    df["BB_Lower"] = sma - 2 * std

    df["BB_Width"] = np.where(
        sma != 0,
        (df["BB_Upper"] - df["BB_Lower"]) / sma,
        0
    )

    # =====================================
    # Returns
    # =====================================
    df["Daily_Return"] = df["Close"].pct_change()

    df["Return_1"] = df["Close"].pct_change(1)
    df["Return_3"] = df["Close"].pct_change(3)
    df["Return_5"] = df["Close"].pct_change(5)
    df["Return_10"] = df["Close"].pct_change(10)

    # =====================================
    # Volatility
    # =====================================
    df["Volatility_5"] = df["Daily_Return"].rolling(5).std()
    df["Volatility_10"] = df["Daily_Return"].rolling(10).std()

    # =====================================
    # Volume Features
    # =====================================
    df["Volume_Change"] = df["Volume"].pct_change()

    # =====================================
    # Clean Data
    # =====================================
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    logger.info("Advanced indicators added successfully.")

    return df