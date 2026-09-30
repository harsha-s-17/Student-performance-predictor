import pandas as pd

from src.data_preprocessing import DATA_PATH, ensure_dataset, load_dataset
from src.prediction import performance_category, predict_dataframe
from src.train_models import train_and_save


def test_prediction_and_category(tmp_path):
    ensure_dataset(DATA_PATH)
    data = load_dataset(DATA_PATH)
    train_and_save(DATA_PATH, tmp_path)
    result = predict_dataframe(data.drop(columns=["G3"]).head(2), tmp_path)
    assert len(result) == 2
    assert performance_category(16) == "Excellent"


def test_unknown_category_is_supported(tmp_path):
    ensure_dataset(DATA_PATH)
    data = load_dataset(DATA_PATH)
    train_and_save(DATA_PATH, tmp_path)
    row = data.drop(columns=["G3"]).head(1).copy()
    row["school"] = "UNKNOWN"
    assert len(predict_dataframe(row, tmp_path)) == 1
