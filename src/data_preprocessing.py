from pathlib import Path
from urllib.request import urlopen

import pandas as pd


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "student-mat.csv"
TARGET = "G3"


def ensure_dataset(data_path: Path = DATA_PATH) -> Path:
    """Download the official UCI Math CSV when it is not present locally."""
    data_path = Path(data_path)
    if data_path.exists():
        return data_path
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(DATA_URL, timeout=30) as response:
        archive = response.read()
    import io
    import zipfile
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        with zipped.open("student-mat.csv") as source:
            data_path.write_bytes(source.read())
    return data_path


def load_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    data_path = Path(data_path)
    if not data_path.exists():
        ensure_dataset(data_path)
    return pd.read_csv(data_path, sep=";")


def split_features_target(data: pd.DataFrame):
    if TARGET not in data.columns:
        raise ValueError(f"Dataset must contain target column '{TARGET}'.")
    return data.drop(columns=[TARGET]), data[TARGET]


def build_preprocessor(features: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(exclude="number").columns.tolist()
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)], remainder="drop")
