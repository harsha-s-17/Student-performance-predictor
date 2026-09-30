import pandas as pd
import pytest

from src.data_preprocessing import build_preprocessor, split_features_target


def sample_data():
    return pd.DataFrame({"studytime": [1, 2, 3, 4], "absences": [2, 4, 0, 8], "sex": ["F", "M", "F", "M"], "G3": [8, 12, 15, 18]})


def test_split_features_target():
    x, y = split_features_target(sample_data())
    assert "G3" not in x.columns
    assert y.tolist() == [8, 12, 15, 18]


def test_preprocessor_encodes_and_preserves_rows():
    data = sample_data()
    x, _ = split_features_target(data)
    transformed = build_preprocessor(x).fit_transform(x)
    assert transformed.shape[0] == len(data)
    assert transformed.shape[1] > len(x.select_dtypes(include="number").columns)


def test_missing_target_is_invalid():
    with pytest.raises(ValueError):
        split_features_target(pd.DataFrame({"studytime": [1]}))
