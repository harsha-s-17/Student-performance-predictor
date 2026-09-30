from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data_preprocessing import DATA_PATH, TARGET, ensure_dataset, load_dataset
from src.eda import (
    correlation_figure,
    score_distribution_figure,
    studytime_figure,
    categorical_average_figure,
    numeric_relationship_figure,
)
from src.prediction import (
    MODEL_DIR,
    ensure_artifacts,
    performance_category,
    predict_dataframe,
)
from src.train_models import load_metrics


st.set_page_config(page_title="Student Performance Analysis", page_icon="📚", layout="wide")


@st.cache_data
def get_data():
    ensure_dataset(DATA_PATH)
    return load_dataset(DATA_PATH)


@st.cache_resource
def get_artifacts():
    ensure_artifacts(DATA_PATH, MODEL_DIR)
    return True


def show_figure(figure):
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)


def render_home(data):
    st.title("Student Performance Prediction and Academic Analysis")
    st.write(
        "A simple machine learning application using the UCI Student Performance "
        "dataset to analyze and predict a student's final math grade."
    )
    st.subheader("How the system works")
    st.write("The app loads the public dataset, cleans and encodes its features, trains Linear Regression and Random Forest Regression models, evaluates them, and applies the trained transformations to new student records.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Students", len(data))
    c2.metric("Input features", len(data.columns) - 1)
    c3.metric("Target", TARGET)
    st.caption("Dataset: UCI Student Performance, Math course. Categories shown in this app are project-defined and are not official grading standards.")


def render_analysis(data):
    st.title("Dataset Analysis")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", data.shape[0])
    c2.metric("Columns", data.shape[1])
    c3.metric("Missing values", int(data.isna().sum().sum()))
    st.subheader("Dataset preview")
    st.dataframe(data.head(10), use_container_width=True)
    st.subheader("Basic statistics")
    st.dataframe(data.describe(include="all").transpose(), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        show_figure(score_distribution_figure(data))
    with col2:
        show_figure(studytime_figure(data))
    col3, col4 = st.columns(2)
    with col3:
        show_figure(numeric_relationship_figure(data, "absences", TARGET, "Absences vs final grade"))
    with col4:
        show_figure(categorical_average_figure(data, "sex", TARGET, "Average final grade by gender"))
    st.subheader("Correlation heatmap")
    show_figure(correlation_figure(data))


def render_model_performance(data):
    st.title("Model Performance")
    get_artifacts()
    metrics = load_metrics(MODEL_DIR)
    st.dataframe(pd.DataFrame(metrics).T, use_container_width=True)
    st.caption("MAE and RMSE are measured in grade points. R² is the coefficient of determination. Metrics are calculated from the trained test predictions.")
    st.subheader("Model comparison")
    metric_df = pd.DataFrame(metrics).T
    st.bar_chart(metric_df[["MAE", "RMSE"]])
    st.subheader("Actual versus predicted final grades")
    predictions = pd.read_csv(MODEL_DIR / "test_predictions.csv")
    chart_df = predictions.set_index("actual")[["Linear Regression", "Random Forest"]]
    st.scatter_chart(chart_df)
    st.subheader("Random Forest feature importance")
    importance = pd.read_csv(MODEL_DIR / "feature_importance.csv").sort_values("importance")
    st.bar_chart(importance.set_index("feature").tail(15))
    st.caption("Feature importance indicates how useful a feature was to this model; it does not prove causation.")


def render_single_prediction(data):
    st.title("Student Prediction")
    get_artifacts()
    st.write("Enter values using the same fields and value formats as the UCI dataset.")
    input_values = {}
    columns = [c for c in data.columns if c != TARGET]
    left, right = st.columns(2)
    for index, column in enumerate(columns):
        series = data[column]
        container = left if index % 2 == 0 else right
        if pd.api.types.is_numeric_dtype(series):
            input_values[column] = container.number_input(column, min_value=float(series.min()), max_value=float(series.max()), value=float(series.median()), step=1.0)
        else:
            input_values[column] = container.selectbox(column, sorted(series.dropna().astype(str).unique()))
    if st.button("Predict Performance", type="primary"):
        score = float(predict_dataframe(pd.DataFrame([input_values]), MODEL_DIR)[0])
        st.metric("Predicted final grade (G3)", f"{score:.2f} / 20")
        st.success(performance_category(score, max_score=20))


def render_batch_prediction(data):
    st.title("Batch Prediction")
    get_artifacts()
    st.write("Upload a CSV containing all input columns from the dataset. The `G3` target column is optional and will be ignored if present.")
    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded is None:
        return
    batch = pd.read_csv(uploaded)
    required = [c for c in data.columns if c != TARGET]
    missing = sorted(set(required) - set(batch.columns))
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        return
    predictions = predict_dataframe(batch[required], MODEL_DIR)
    result = batch.copy()
    result["predicted_score"] = predictions.round(2)
    result["performance_category"] = [performance_category(x, 20) for x in predictions]
    st.dataframe(result, use_container_width=True)
    st.download_button("Download predictions CSV", result.to_csv(index=False).encode("utf-8"), "student_predictions.csv", "text/csv")


def main():
    data = get_data()
    pages = {"Home": render_home, "Dataset Analysis": render_analysis, "Model Performance": render_model_performance, "Student Prediction": render_single_prediction, "Batch Prediction": render_batch_prediction}
    page = st.sidebar.radio("Navigate", list(pages))
    pages[page](data)
    st.sidebar.divider()
    st.sidebar.caption("About\n\nModels: Linear Regression and Random Forest Regression\n\nSource: UCI Machine Learning Repository")


if __name__ == "__main__":
    main()
