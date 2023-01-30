from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data import generate_synthetic, load_csv, train_test_split_dataset
from src.detector import OutlierDetector
from src.evaluate import evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Outlier Detection System")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to CSV input data. If omitted, synthetic data is generated.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=None,
        help="Optional ground-truth label column in CSV (0=inlier, 1=outlier).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="isolation_forest",
        choices=["isolation_forest", "local_outlier_factor"],
        help="Outlier detection model.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.1,
        help="Expected outlier proportion used by the model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions.csv"),
        help="Where predictions CSV is saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input:
        dataset = load_csv(args.input, args.label_column)
    else:
        dataset = generate_synthetic()

    train_ds, test_ds = train_test_split_dataset(dataset, test_size=0.25)

    detector = OutlierDetector(
        model_type=args.model,
        contamination=args.contamination,
        random_state=42,
    )
    detector.fit(train_ds.features)
    result = detector.predict(test_ds.features)

    output_df = pd.DataFrame(test_ds.features.copy())
    output_df["predicted_outlier"] = result.predictions
    output_df["anomaly_score"] = result.scores
    output_df.to_csv(args.output, index=False)

    print(f"Saved predictions to: {args.output}")
    print(f"Detected outliers: {int(result.predictions.sum())}/{len(result.predictions)}")

    metrics = evaluate_predictions(
        y_true=test_ds.labels.to_numpy() if test_ds.labels is not None else None,
        y_pred=result.predictions,
    )
    if metrics is None:
        print("No labels provided. Skipping evaluation metrics.")
        return

    print("\nEvaluation")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall:    {metrics.recall:.4f}")
    print(f"F1 Score:  {metrics.f1:.4f}")
    print("Confusion Matrix:")
    print(metrics.confusion_matrix)
    print("\nClassification Report:")
    print(metrics.report)


if __name__ == "__main__":
    main()

