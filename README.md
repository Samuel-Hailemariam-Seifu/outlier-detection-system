# Outlier Detection System

Simple Python project for detecting anomalies (outliers) in tabular numeric data.

## Features
- Supports two models:
  - Isolation Forest
  - Local Outlier Factor (novelty mode)
- Works with:
  - synthetic generated data (default)
  - your own CSV files
- Optional evaluation if your CSV has a ground-truth label column.
- Exports predictions and anomaly scores to CSV.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### UI App (Streamlit)
```bash
streamlit run src/ui.py
```

### 1) Run with synthetic data
```bash
python -m src.main
```

### 2) Run with custom CSV
```bash
python -m src.main --input data.csv
```

### 3) Run with labels for evaluation
Assumes `is_outlier` is 0 (inlier) / 1 (outlier):
```bash
python -m src.main --input data.csv --label-column is_outlier
```

### 4) Switch model
```bash
python -m src.main --model local_outlier_factor
```

### 5) Customize expected anomaly rate
```bash
python -m src.main --contamination 0.05
```

## Output
By default, output file is `predictions.csv` with:
- original feature columns
- `predicted_outlier` (0 or 1)
- `anomaly_score` (higher means more anomalous)

