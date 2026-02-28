import numpy as np
from src.logger import logger

def backtest_strategy(df, model):

    logger.info("Running backtest...")

    X = df.drop("label", axis=1)
    X = X.select_dtypes(include=["number"])

    df["prediction"] = model.predict(X)

    # Long-only strategy
    df["position"] = 0
    df.loc[df["prediction"] == 1, "position"] = 1

    # Forward fill position correctly (new safe way)
    df["position"] = df["position"].ffill()

    df["strategy_return"] = df["position"].shift(1) * df["Daily_Return"]

    df["cumulative_strategy"] = (1 + df["strategy_return"]).cumprod()

    total_return = df["cumulative_strategy"].iloc[-1] - 1
    sharpe = (
        df["strategy_return"].mean() /
        df["strategy_return"].std()
    ) * np.sqrt(252)

    metrics = {
        "Total Return": round(total_return, 3),
        "Sharpe Ratio": round(sharpe, 3)
    }

    logger.info(f"Backtest Total Return: {total_return:.2%}")
    logger.info(f"Sharpe Ratio: {sharpe:.2f}")

    return df, metrics