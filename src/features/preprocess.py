"""Preprocessing utilities for AQI modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessedData:
    """Container for train/validation splits and fitted scaler."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    scaler: StandardScaler



def preprocess_data(
    df: pd.DataFrame,
    target_column: str = "AQI",
    feature_columns: Iterable[str] | None = None,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> PreprocessedData:
    """Prepare features/labels with null handling, scaling, and split.

    - Fills nulls in numeric columns with column medians.
    - Splits into train/validation sets.
    - Fits ``StandardScaler`` on training features and transforms both sets.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    feature_columns = list(feature_columns)

    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature column(s): {', '.join(missing_features)}")

    model_df = df[feature_columns + [target_column]].copy()

    numeric_cols = model_df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        model_df[col] = model_df[col].fillna(model_df[col].median())

    X = model_df[feature_columns]
    y = model_df[target_column]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_columns,
        index=X_train.index,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=feature_columns,
        index=X_val.index,
    )

    return PreprocessedData(
        X_train=X_train_scaled,
        X_val=X_val_scaled,
        y_train=y_train,
        y_val=y_val,
        scaler=scaler,
    )
