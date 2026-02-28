import os
from src.data_loader import fetch_data
from src.data_preprocessing import clean_data
from src.features import add_technical_indicators
from src.labels import create_labels
from src.model import train_model
from src.evaluation import evaluate
from src.backtest import backtest_strategy
from src.visualization import plot_strategy
from src.config import PROCESSED_DATA_PATH
from src.logger import logger


def main():

    logger.info("========== ML Trading Pipeline Started ==========")

    # STEP 1
    logger.info("STEP 1: Fetching data...")
    df = fetch_data()

    # STEP 2
    logger.info("STEP 2: Cleaning data...")
    df = clean_data(df)

    # STEP 3
    logger.info("STEP 3: Adding technical indicators...")
    df = add_technical_indicators(df)

    # STEP 4
    logger.info("STEP 4: Creating labels...")
    df = create_labels(df)

    # Save processed dataset
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH)
    logger.info(f"Processed dataset saved to {PROCESSED_DATA_PATH}")

    # STEP 5
    logger.info("STEP 5: Training model...")
    model, X_test, y_test = train_model(df)

    # STEP 6
    logger.info("STEP 6: Evaluating model...")
    evaluate(model, X_test, y_test)

    # STEP 7
    logger.info("STEP 7: Running backtest...")
    backtest_df, metrics = backtest_strategy(df, model)

    logger.info("Backtest Metrics:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")

    # STEP 9
    logger.info("STEP 9: Plotting results...")
    plot_strategy(backtest_df)

    logger.info("========== Pipeline Completed Successfully ==========")


if __name__ == "__main__":
    main()