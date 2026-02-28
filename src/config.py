SYMBOL = "^BSESN"   # Change to ^NSEI if needed
START_DATE = "2015-01-01"
END_DATE = None
INTERVAL = "1d"

# Labeling
FUTURE_WINDOW = 5
BUY_THRESHOLD = 0.005
SELL_THRESHOLD = -0.005

# Model
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 300
MAX_DEPTH = 8

MODEL_PATH = "models/random_forest.pkl"

# Paths
RAW_DATA_PATH = "data/raw/sensex.csv"
PROCESSED_DATA_PATH = "data/processed/features.csv"
MODEL_PATH = "models/random_forest.pkl"