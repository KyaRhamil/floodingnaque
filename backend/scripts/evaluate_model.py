import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Resolve paths relative to backend directory
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
MODELS_DIR = BACKEND_DIR / "models"
DATA_DIR = BACKEND_DIR / "data"


def evaluate_model(model_path=None, data_path=None):
    """Evaluate the trained model and generate metrics."""

    # Use default paths relative to backend directory
    if model_path is None:
        model_path = MODELS_DIR / "flood_rf_model.joblib"
    else:
        model_path = Path(model_path)

    if data_path is None:
        data_path = DATA_DIR / "synthetic_dataset.csv"
    else:
        data_path = Path(data_path)

    # Load model and data
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    model = joblib.load(model_path)
    data = pd.read_csv(data_path)

    X = data.drop("flood", axis=1)
    y = data["flood"]

    # Predict
    y_pred = model.predict(X)

    # Accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy}")  # OK: has curly braces

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(MODELS_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Feature Importance
    if hasattr(model, "feature_importances_"):
        plt.figure(figsize=(10, 6))
        feature_importances = pd.Series(model.feature_importances_, index=X.columns)
        feature_importances.nlargest(10).plot(kind="barh")
        plt.title("Top 10 Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(MODELS_DIR / "feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("Evaluation complete.")
    return accuracy


if __name__ == "__main__":
    evaluate_model()
