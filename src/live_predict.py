import yfinance as yf
import joblib
from src.config import SYMBOL, MODEL_PATH
from src.features import add_technical_indicators
from src.data_preprocessing import clean_data

def live_prediction(interval="5m", period="5d"):

    df = yf.download(SYMBOL, interval=interval, period=period)
    df.reset_index(inplace=True)
    df = clean_data(df)
    df = add_technical_indicators(df)

    latest = df.iloc[-1:]

    model = joblib.load(MODEL_PATH)

    prediction = model.predict(latest)[0]
    probabilities = model.predict_proba(latest)[0]

    signal_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}

    return {
        "Signal": signal_map[prediction],
        "Confidence": round(max(probabilities), 3),
        "Timeframe": interval
    }