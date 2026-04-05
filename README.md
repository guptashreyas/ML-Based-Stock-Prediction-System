#  ML-Based Stock Prediction System

A machine learning trading system that predicts BUY / HOLD / SELL signals using historical market data and technical indicators.

This repository walks through a full pipeline:
1. Download market data
2. Clean and transform it
3. Create technical-indicator features
4. Generate labels for supervised learning
5. Train an ML model (XGBoost)
6. Evaluate predictions
7. Backtest a simple strategy
8. Run a Streamlit app for live signal generation

---
##  Project Structure

```text
ML-Based-Stock-Prediction-System/
├── app.py                     # Streamlit app (interactive signal dashboard)
├── main.py                    # End-to-end training + evaluation + backtest pipeline
├── run_live.py                # Simple script to run live prediction in terminal
├── requirements.txt           # Python dependencies
├── src/
   ├── config.py              # Symbol, date range, labeling/model/path settings
   ├── data_loader.py         # Downloads historical data from Yahoo Finance
   ├── data_preprocessing.py  # Sorting, deduplication, NA handling
   ├── features.py            # Technical indicators and engineered features
   ├── labels.py              # Creates BUY/HOLD/SELL class labels
   ├── model.py               # Trains and saves XGBoost model + artifacts
   ├── evaluation.py          # Classification report
   ├── backtest.py            # Strategy simulation and metrics
   ├── visualization.py       # Plotly chart for signals and EMAs
   ├── live_predict.py        # Inference helper for latest data
   └── logger.py              # Logging setup

```

---
##  How the pipeline works 

### 1) Data download
`src/data_loader.py` downloads historical prices (Open, High, Low, Close, Volume) using `yfinance`.

### 2) Data cleaning
`src/data_preprocessing.py` sorts by date, removes duplicates/nulls, and sets `Date` as index.

### 3) Feature engineering
`src/features.py` adds common technical indicators:
- These indicators are math-based signals built from historical price and volume data to help the model detect trend, momentum, volatility, and market strength patterns.
- EMA (20, 50), EMA spread
- RSI
- MACD + MACD signal
- Bollinger Bands (+ width)
- Returns (1, 3, 5, 10 periods)
- Volatility features
- Volume change

### 4) Label creation
`src/labels.py` creates labels based on future returns over `FUTURE_WINDOW` periods:
- **1 → BUY**
- **0 → HOLD**
- **-1 → SELL**

It uses percentile cutoffs to make classes more balanced.

### 5) Model training
`src/model.py` trains an `XGBClassifier` and saves:
- model file
- feature column list
- label map for converting model class IDs back to trading signals

### 6) Evaluation
`src/evaluation.py` prints a classification report on the test split.

### 7) Backtesting
`src/backtest.py` applies predictions to a simple position strategy and reports:
- Total return
- Sharpe ratio
- Max drawdown
- Win rate
- Number of trades

### 8) Visualization
`src/visualization.py` draws candlesticks, EMA lines, and buy/sell markers using Plotly.

---

##  Setup Instructions

### Prerequisites
- Python 3.10+ (3.11 recommended)
- pip

### 1) Clone the repository
```bash
python main.py
git clone https://github.com/guptashreyas/ML-Based-Stock-Prediction-System.git
cd ML-Based-Stock-Prediction-System
```

### 2) Create and activate a virtual environment (recommended)

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate
```

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

## Run Streamlit 
### 3) Install dependencies
```bash
pip install -r requirements.txt
```

---

##  How to run

### A) Full training + backtest pipeline
```bash
python main.py
```

Expected outputs after successful run:
- Processed data CSV in `data/processed/`
- Model artifacts in `models/`
- Logs in `logs/app.log`
- Console metrics + classification report

### B) Streamlit dashboard (interactive)
```bash
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

### C) Terminal-based live prediction script
```bash
python run_live.py
```

---

## 🔧 Configuration

Edit `src/config.py` to customize:
- Symbol (default example: `^BSESN`)
- Date range
- Label horizon (`FUTURE_WINDOW`)
- Thresholds/hyperparameters
- File paths
---

## 📄 License

**This project is licensed under the MIT License.**
---


