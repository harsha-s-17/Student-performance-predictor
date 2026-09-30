# Student Performance Prediction and Academic Analysis

Beginner-friendly B.Tech CSE internship project using the UCI Student Performance dataset for the Math course. It uses only Linear Regression and Random Forest Regression.

## Dataset

Official UCI page: https://archive.ics.uci.edu/dataset/320/student%2Bperformance

The application downloads `student-mat.csv` automatically on first run from the official UCI archive. The target is `G3`, the final math grade from 0 to 20.

## Run locally

1. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Start the application:

   ```powershell
   streamlit run app.py
   ```

The browser page will open at the local Streamlit address, usually `http://localhost:8501`.

## Tests

```powershell
pytest
```

Model files and downloaded data are generated locally and excluded from Git. The first model-related page or prediction triggers training automatically; to train explicitly, run `python -m src.train_models`.

## Project structure

```text
ML Project/
├── app.py
├── requirements.txt
├── README.md
├── data/                 # downloaded student-mat.csv
├── models/               # generated models, metrics, and charts data
├── src/
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── prediction.py
│   └── train_models.py
└── tests/
    ├── test_models.py
    ├── test_prediction.py
    └── test_preprocessing.py
```
