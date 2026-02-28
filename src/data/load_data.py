"""Data loading utilities for AQI forecasting datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_REQUIRED_COLUMNS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]


def load_aqi_data(csv_path: str | Path, required_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Load an AQI dataset from CSV and validate required columns.

    Args:
        csv_path: Path to the source CSV file.
        required_columns: Columns expected in the dataset. If omitted,
            ``DEFAULT_REQUIRED_COLUMNS`` is used.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If required columns are missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    expected = list(required_columns) if required_columns is not None else DEFAULT_REQUIRED_COLUMNS
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required column(s): "
            f"{', '.join(missing)}. Available columns: {', '.join(df.columns)}"
        )

    return df
