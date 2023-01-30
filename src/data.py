from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.datasets import make_blobs


@dataclass
class Dataset:
    features: pd.DataFrame
    labels: Optional[pd.Series] = None


def load_csv(path: Path, label_column: Optional[str] = None) -> Dataset:
    df = pd.read_csv(path)

    labels = None
    if label_column:
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in {path}.")
        labels = df[label_column].astype(int)
        df = df.drop(columns=[label_column])

    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        raise ValueError("No numeric columns found. Outlier models require numeric features.")

    return Dataset(features=numeric_df, labels=labels)


def generate_synthetic(
    n_inliers: int = 900,
    n_outliers: int = 100,
    n_features: int = 2,
    random_state: int = 42,
) -> Dataset:
    inliers, _ = make_blobs(
        n_samples=n_inliers,
        n_features=n_features,
        centers=1,
        cluster_std=1.0,
        random_state=random_state,
    )
    outliers, _ = make_blobs(
        n_samples=n_outliers,
        n_features=n_features,
        centers=1,
        cluster_std=5.0,
        center_box=(10.0, 15.0),
        random_state=random_state + 1,
    )

    features = pd.DataFrame(
        data=pd.concat(
            [pd.DataFrame(inliers), pd.DataFrame(outliers)],
            ignore_index=True,
        )
    )
    features.columns = [f"feature_{i}" for i in range(features.shape[1])]

    labels = pd.Series([0] * n_inliers + [1] * n_outliers, name="is_outlier")
    return Dataset(features=features, labels=labels)


def train_test_split_dataset(dataset: Dataset, test_size: float = 0.25) -> Tuple[Dataset, Dataset]:
    split_idx = int((1.0 - test_size) * len(dataset.features))
    train_x = dataset.features.iloc[:split_idx].reset_index(drop=True)
    test_x = dataset.features.iloc[split_idx:].reset_index(drop=True)

    if dataset.labels is None:
        return Dataset(train_x, None), Dataset(test_x, None)

    train_y = dataset.labels.iloc[:split_idx].reset_index(drop=True)
    test_y = dataset.labels.iloc[split_idx:].reset_index(drop=True)
    return Dataset(train_x, train_y), Dataset(test_x, test_y)

