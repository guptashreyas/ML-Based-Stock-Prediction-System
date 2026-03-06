import numpy as np
import joblib
from src.logger import logger


def backtest_strategy(df, model):

    logger.info("Running advanced backtest...")

    # Load saved artifacts
    feature_columns = joblib.load("models/feature_columns2.pkl")
    reverse_label_map = joblib.load("models/label_map.pkl")

    # Ensure features match training exactly
    X = df[feature_columns].copy()

    # Predictions
    probs = model.predict_proba(X)
    raw_preds = model.predict(X)

    preds = np.array([reverse_label_map[p] for p in raw_preds])
    confidence = probs.max(axis=1)

    # Confidence filtering
    threshold = 0.6
    preds[confidence < threshold] = 0

    df["prediction"] = preds
    df["confidence"] = confidence

    # Position logic
    df["position"] = 0
    df.loc[df["prediction"] == 1, "position"] = 1
    df.loc[df["prediction"] == -1, "position"] = 0

    df["position"] = df["position"].ffill().fillna(0)

    # Transaction cost
    transaction_cost = 0.0005
    df["trade"] = df["position"].diff().abs()

    df["strategy_return"] = (
        df["position"].shift(1) * df["Daily_Return"]
        - df["trade"] * transaction_cost
    )

    df["strategy_return"].fillna(0, inplace=True)

    # Metrics
    df["cumulative_strategy"] = (1 + df["strategy_return"]).cumprod()

    total_return = df["cumulative_strategy"].iloc[-1] - 1

    sharpe = (
        df["strategy_return"].mean()
        / df["strategy_return"].std()
    ) * np.sqrt(252)

    rolling_max = df["cumulative_strategy"].cummax()
    drawdown = df["cumulative_strategy"] / rolling_max - 1
    max_drawdown = drawdown.min()

    trades = int(df["trade"].sum())

    win_rate = (
        len(df[df["strategy_return"] > 0])
        / len(df[df["trade"] > 0])
        if len(df[df["trade"] > 0]) > 0 else 0
    )

    metrics = {
        "Total Return": round(total_return, 3),
        "Sharpe Ratio": round(sharpe, 3),
        "Max Drawdown": round(max_drawdown, 3),
        "Win Rate": round(win_rate, 3),
        "Trades": trades
    }

    logger.info(f"Total Return: {total_return:.2%}")
    logger.info(f"Sharpe Ratio: {sharpe:.2f}")
    logger.info(f"Max Drawdown: {max_drawdown:.2%}")
    logger.info(f"Win Rate: {win_rate:.2%}")
    logger.info(f"Total Trades: {trades}")

    return df, metrics