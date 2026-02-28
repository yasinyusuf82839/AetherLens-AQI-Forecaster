from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline AQI forecaster and emit evaluation artifacts.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/aqi_training.csv"),
        help="Path to training data CSV.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="AQI",
        help="Name of target column in dataset.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where metrics and feature importance artifacts are written.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splits and model behavior.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    if not args.data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{args.data_path}'. Use --data-path to provide a CSV file."
        )

    df = pd.read_csv(args.data_path)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in dataset columns: {list(df.columns)}")

    X = df.drop(columns=[args.target])
    y = df[args.target]

    # Minimal preprocessing for mixed-type tabular data.
    X = pd.get_dummies(X, dummy_na=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.validation_size,
        random_state=args.seed,
        shuffle=True,
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    mae = float(mean_absolute_error(y_val, preds))
    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    r2 = float(r2_score(y_val, preds))

    print("Validation metrics")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "split": "validation",
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "seed": args.seed,
        "validation_size": args.validation_size,
        "data_path": str(args.data_path),
        "target": args.target,
    }

    metrics_path = args.artifacts_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    fi_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    fi_path = args.artifacts_dir / "feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved feature importance: {fi_path}")


if __name__ == "__main__":
    main()
