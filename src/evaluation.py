from sklearn.metrics import classification_report
from src.logger import logger

def evaluate(model, X_test, y_test):

    logger.info("Evaluating model...")

    preds = model.predict(X_test)

    report = classification_report(y_test, preds)

    print("\nClassification Report:\n")
    print(report)

    logger.info("Evaluation completed.")