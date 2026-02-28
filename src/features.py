import numpy as np
from src.logger import logger

def add_technical_indicators(df):

    logger.info("Adding technical indicators...")

    # EMA
    df["EMA_20"] = df["Close"].ewm(span=20).mean()
    df["EMA_50"] = df["Close"].ewm(span=50).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = df["Close"].ewm(span=12).mean()
    ema_slow = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()

    # Bollinger Bands
    sma = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()

    df["BB_Upper"] = sma + 2 * std
    df["BB_Lower"] = sma - 2 * std

    # 🔥 Safe BB Width calculation
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / sma

    # Returns
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volume_Change"] = df["Volume"].pct_change()

    # 🔥 CRITICAL FIX
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    logger.info("Technical indicators added successfully.")

    return df