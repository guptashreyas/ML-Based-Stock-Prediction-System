from src.config import FUTURE_WINDOW, BUY_THRESHOLD, SELL_THRESHOLD
from src.logger import logger

def create_labels(df):

    logger.info("Creating BUY/HOLD/SELL labels...")
    
    future_return = (
        df["Close"].shift(-FUTURE_WINDOW) - df["Close"]
    ) / df["Close"]

    df["label"] = 0  # HOLD
    logger.info(df["label"].value_counts())
    

    df.loc[future_return >= BUY_THRESHOLD, "label"] = 1
    df.loc[future_return <= SELL_THRESHOLD, "label"] = -1

    df = df.dropna()
    print(df["label"].value_counts())

    logger.info("Labels created successfully.")

    return df