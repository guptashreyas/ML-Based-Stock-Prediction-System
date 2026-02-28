from src.logger import logger

def clean_data(df):

    logger.info("Cleaning data...")

    df = df.sort_values("Date")
    df = df.drop_duplicates()
    df = df.dropna()

    df.set_index("Date", inplace=True)

    logger.info(f"Data after cleaning: {len(df)} rows")

    return df