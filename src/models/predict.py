"""Prediction utilities for AQI forecasting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def predict_aqi(model_path: str | Path, features: dict[str, Any]) -> float:
    """Load model artifact and predict AQI for one feature row."""
    artifact = joblib.load(model_path)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_columns = artifact["feature_columns"]

    missing = [f for f in feature_columns if f not in features]
    if missing:
        raise ValueError(f"Missing feature input(s): {', '.join(missing)}")

    row = pd.DataFrame([features], columns=feature_columns)
    row = row.replace([np.inf, -np.inf], np.nan)
    for col in feature_columns:
        if row[col].isna().any():
            row[col] = row[col].fillna(0)

    scaled = scaler.transform(row)
    pred = model.predict(scaled)[0]
    return float(pred)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict AQI using a trained model artifact.")
    parser.add_argument(
        "--model-path",
        default="artifacts/random_forest_aqi.joblib",
        help="Path to trained model artifact",
    )
    parser.add_argument(
        "--features-json",
        required=True,
        help=(
            "JSON string with pollutant features, e.g. "
            '\'{"PM2.5": 35, "PM10": 70, "NO2": 22, "SO2": 8, "CO": 0.8, "O3": 40}\''
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    features = json.loads(args.features_json)
    prediction = predict_aqi(args.model_path, features)
    print(f"Predicted AQI: {prediction:.2f}")


if __name__ == "__main__":
    main()
