from xgboost import XGBClassifier
import joblib
import os
from sklearn.model_selection import train_test_split
from src.logger import logger


def train_model(df):

    logger.info("Training XGBoost model...")

    # Map labels for XGBoost
    label_map = {-1: 0, 0: 1, 1: 2}
    reverse_label_map = {0: -1, 1: 0, 2: 1}

    y = df["label"].map(label_map)

    X = df.drop("label", axis=1)
    X = X.select_dtypes(include=["number"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )

    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/xgb_model.pkl")
    joblib.dump(X.columns.tolist(), "models/feature_columns2.pkl")
    joblib.dump(reverse_label_map, "models/label_map.pkl")

    logger.info("XGBoost model trained and saved.")

    return model, X_test, y_test