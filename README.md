# 📈 ML-Based Buy / Hold / Sell Prediction System

A machine learning trading system that predicts BUY / HOLD / SELL signals using historical market data and technical indicators.

The project builds an end-to-end ML pipeline including data collection, feature engineering, model training, backtesting, and a Streamlit dashboard for live predictions.

## Features
- Multi-class classification (BUY / HOLD / SELL)
- Advanced technical indicators:
    -EMA (20, 50)
    -RSI
    -MACD
    -Bollinger Bands
    -Volatility & return features
- XGBoost model for prediction
- Live prediction dashboard via Streamlit

## Tech Stack
- Python 3.11
- scikit-learn
- xgboost
- yfinance
- Streamlit
- Plotly
-Pandas / NumPy

## Run Training Pipeline

```bash
python main.py
```

## Run Streamlit 


```bash
streamlit run app.py
```

## Data source: 
Market data is fetched from Yahoo Finance using the yfinance API.
## Data source: Yahoo Finance 
