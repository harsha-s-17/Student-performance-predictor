import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .data_preprocessing import build_preprocessor, load_dataset, split_features_target


RANDOM_STATE = 42


def train_and_save(data_path, model_dir):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset(data_path)
    features, target = split_features_target(data)
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=RANDOM_STATE)
    preprocessor = build_preprocessor(x_train)
    x_train_ready = preprocessor.fit_transform(x_train)
    x_test_ready = preprocessor.transform(x_test)
    models = {"Linear Regression": LinearRegression(), "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)}
    metrics = {}
    predictions = {"actual": y_test.to_numpy()}
    for name, model in models.items():
        model.fit(x_train_ready, y_train)
        predicted = model.predict(x_test_ready)
        metrics[name] = {"MAE": float(mean_absolute_error(y_test, predicted)), "RMSE": float(np.sqrt(mean_squared_error(y_test, predicted))), "R2": float(r2_score(y_test, predicted))}
        predictions[name] = predicted
        joblib.dump(model, model_dir / ("linear_regression.pkl" if name == "Linear Regression" else "random_forest.pkl"))
    joblib.dump(preprocessor, model_dir / "preprocessor.pkl")
    pd.DataFrame(predictions).to_csv(model_dir / "test_predictions.csv", index=False)
    names = preprocessor.get_feature_names_out()
    pd.DataFrame({"feature": names, "importance": models["Random Forest"].feature_importances_}).sort_values("importance", ascending=False).to_csv(model_dir / "feature_importance.csv", index=False)
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_metrics(model_dir):
    return json.loads((Path(model_dir) / "metrics.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    from .data_preprocessing import DATA_PATH
    train_and_save(DATA_PATH, Path(__file__).resolve().parents[1] / "models")
