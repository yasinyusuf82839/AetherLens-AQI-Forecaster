"""Model training pipeline for AQI forecasting."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.load_data import DEFAULT_REQUIRED_COLUMNS, load_aqi_data
from src.features.preprocess import preprocess_data


def train_model(
    data_path: str | Path,
    model_out: str | Path = "artifacts/random_forest_aqi.joblib",
    random_state: int = 42,
    n_estimators: int = 200,
) -> Path:
    """Train baseline RandomForest model and persist artifacts."""
    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    df = load_aqi_data(data_path, required_columns=DEFAULT_REQUIRED_COLUMNS)

    features = [column for column in DEFAULT_REQUIRED_COLUMNS if column != "AQI"]
    processed = preprocess_data(df, target_column="AQI", feature_columns=features)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(processed.X_train, processed.y_train)

    preds = model.predict(processed.X_val)
    rmse = mean_squared_error(processed.y_val, preds) ** 0.5
    mae = mean_absolute_error(processed.y_val, preds)
    r2 = r2_score(processed.y_val, preds)

    print("Validation metrics")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE : {mae:.3f}")
    print(f"  R^2 : {r2:.3f}")

    bundle = {
        "model": model,
        "scaler": processed.scaler,
        "feature_columns": features,
        "target_column": "AQI",
        "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
    }
    joblib.dump(bundle, model_out)
    print(f"Saved model artifact to: {model_out}")

    return model_out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train baseline AQI prediction model.")
    parser.add_argument("--data-path", required=True, help="Path to training CSV data")
    parser.add_argument(
        "--model-out",
        default="artifacts/random_forest_aqi.joblib",
        help="Output path for saved model artifact",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    train_model(
        data_path=args.data_path,
        model_out=args.model_out,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )


if __name__ == "__main__":
    main()
