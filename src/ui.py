from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data import Dataset, generate_synthetic, train_test_split_dataset
from src.detector import OutlierDetector
from src.evaluate import evaluate_predictions


st.set_page_config(page_title="Outlier Detection System", layout="wide")
st.title("Outlier Detection System")
st.caption("Detect anomalies using Isolation Forest or Local Outlier Factor.")


def _build_dataset_from_upload(file_obj, label_column: str | None) -> Dataset:
    df = pd.read_csv(file_obj)
    labels = None

    if label_column:
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in uploaded CSV.")
        labels = df[label_column].astype(int)
        df = df.drop(columns=[label_column])

    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        raise ValueError("No numeric columns found in the uploaded CSV.")

    return Dataset(features=numeric_df, labels=labels)


with st.sidebar:
    st.header("Configuration")
    source = st.radio("Data source", ["Synthetic", "Upload CSV"], index=0)
    model_type = st.selectbox(
        "Model",
        options=["isolation_forest", "local_outlier_factor"],
        index=0,
    )
    contamination = st.slider("Contamination", min_value=0.01, max_value=0.40, value=0.10, step=0.01)
    test_size = st.slider("Test split", min_value=0.1, max_value=0.5, value=0.25, step=0.05)

    uploaded_file = None
    label_column = None
    if source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        label_column = st.text_input("Label column (optional, 0=inlier, 1=outlier)", value="")
        label_column = label_column.strip() or None

run = st.button("Run Detection", type="primary")

if run:
    try:
        if source == "Synthetic":
            dataset = generate_synthetic()
        else:
            if uploaded_file is None:
                st.warning("Please upload a CSV file first.")
                st.stop()
            dataset = _build_dataset_from_upload(uploaded_file, label_column)

        train_ds, test_ds = train_test_split_dataset(dataset, test_size=test_size)

        detector = OutlierDetector(
            model_type=model_type,
            contamination=contamination,
            random_state=42,
        )
        detector.fit(train_ds.features)
        result = detector.predict(test_ds.features)

        output_df = test_ds.features.copy()
        output_df["predicted_outlier"] = result.predictions
        output_df["anomaly_score"] = result.scores

        detected = int(result.predictions.sum())
        st.success(f"Detected {detected} outliers out of {len(result.predictions)} rows.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows analyzed", f"{len(result.predictions)}")
        c2.metric("Outliers detected", f"{detected}")
        c3.metric("Outlier rate", f"{detected / len(result.predictions):.2%}")

        st.subheader("Results Preview")
        st.dataframe(output_df.head(50), use_container_width=True)

        csv_bytes = output_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download full predictions CSV",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
        )

        if test_ds.labels is not None:
            metrics = evaluate_predictions(
                y_true=test_ds.labels.to_numpy(),
                y_pred=result.predictions,
            )
            if metrics is not None:
                st.subheader("Evaluation")
                m1, m2, m3 = st.columns(3)
                m1.metric("Precision", f"{metrics.precision:.4f}")
                m2.metric("Recall", f"{metrics.recall:.4f}")
                m3.metric("F1 Score", f"{metrics.f1:.4f}")
                st.text("Confusion Matrix")
                st.write(metrics.confusion_matrix)
                st.text("Classification Report")
                st.code(metrics.report)

        if output_df.shape[1] >= 4:
            # Use the first two feature columns for a quick visual scatter.
            feature_cols = [c for c in output_df.columns if c not in ("predicted_outlier", "anomaly_score")]
            if len(feature_cols) >= 2:
                st.subheader("2D Outlier View")
                chart_df = output_df[[feature_cols[0], feature_cols[1], "predicted_outlier"]].copy()
                chart_df["predicted_outlier"] = chart_df["predicted_outlier"].map({0: "Inlier", 1: "Outlier"})
                st.scatter_chart(
                    chart_df,
                    x=feature_cols[0],
                    y=feature_cols[1],
                    color="predicted_outlier",
                )
    except Exception as exc:
        st.error(f"Failed to run detection: {exc}")

