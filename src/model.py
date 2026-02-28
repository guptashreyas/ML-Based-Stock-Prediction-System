import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.config import MODEL_PATH, RANDOM_STATE, N_ESTIMATORS, MAX_DEPTH
from src.logger import logger

def train_model(df):

    logger.info("Training RandomForest model...")

    X = df.drop("label", axis=1)

    # Ensure only numeric columns
    X = X.select_dtypes(include=["number"])
    feature_columns = X.columns.tolist()

    joblib.dump(feature_columns, "models/feature_columns.pkl")
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        class_weight={-1:2, 0:1, 1:2}
    )

    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    logger.info("Model trained and saved.")

    return model, X_test, y_test