import numpy as np
from src.config import FUTURE_WINDOW
from src.logger import logger

def create_labels(df):

    logger.info("Creating percentile-based labels...")

    future_return = (
        df["Close"].shift(-FUTURE_WINDOW) - df["Close"]
    ) / df["Close"]

    # Use percentiles for balanced classes
    upper = future_return.quantile(0.7)
    lower = future_return.quantile(0.3)

    df["label"] = 0

    df.loc[future_return >= upper, "label"] = 1
    df.loc[future_return <= lower, "label"] = -1

    df.dropna(inplace=True)

    logger.info(df["label"].value_counts())

    return df