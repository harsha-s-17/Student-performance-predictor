from src.data_preprocessing import DATA_PATH, ensure_dataset
from src.train_models import train_and_save


def test_model_training_creates_artifacts(tmp_path):
    ensure_dataset(DATA_PATH)
    metrics = train_and_save(DATA_PATH, tmp_path)
    assert set(metrics) == {"Linear Regression", "Random Forest"}
    assert (tmp_path / "linear_regression.pkl").exists()
    assert (tmp_path / "random_forest.pkl").exists()
    assert (tmp_path / "preprocessor.pkl").exists()
