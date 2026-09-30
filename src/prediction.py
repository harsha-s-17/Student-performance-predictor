from pathlib import Path

import joblib

from .data_preprocessing import DATA_PATH
from .train_models import train_and_save


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def ensure_artifacts(data_path=DATA_PATH, model_dir=MODEL_DIR):
    model_dir = Path(model_dir)
    required = ["preprocessor.pkl", "linear_regression.pkl", "random_forest.pkl", "metrics.json", "test_predictions.csv", "feature_importance.csv"]
    if not all((model_dir / name).exists() for name in required):
        train_and_save(data_path, model_dir)


def predict_dataframe(dataframe, model_dir=MODEL_DIR):
    model_dir = Path(model_dir)
    preprocessor = joblib.load(model_dir / "preprocessor.pkl")
    model = joblib.load(model_dir / "random_forest.pkl")
    return model.predict(preprocessor.transform(dataframe))


def performance_category(score, max_score=20):
    percentage = (float(score) / max_score) * 100
    if percentage < 40:
        return "Needs Improvement"
    if percentage < 60:
        return "Average"
    if percentage < 75:
        return "Good"
    return "Excellent"
