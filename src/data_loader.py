import yfinance as yf
import pandas as pd
from src.config import SYMBOL, START_DATE, END_DATE, INTERVAL
from src.logger import logger

def fetch_data():

    logger.info(f"Downloading data for {SYMBOL}")

    df = yf.download(
        SYMBOL,
        start=START_DATE,
        end=END_DATE,
        interval=INTERVAL,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        logger.error("No data returned from Yahoo Finance.")
        raise ValueError("Data download failed.")

    # 🔥 Fix MultiIndex issue
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    logger.info(f"Downloaded {len(df)} rows.")

    return df