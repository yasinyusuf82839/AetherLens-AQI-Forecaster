# AetherLens-AQI-Forecaster

See tomorrow’s air quality today—Python + machine learning that predicts AQI from pollutant signals.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Use Python 3.9+ for best compatibility with modern ML/data tooling.

## Dataset Format and Placement

Expected input data file:

- Path: `data/aqi.csv`
- Format: CSV with a header row
- Typical columns (example):
  - `timestamp` (ISO date/time)
  - `pm2_5`, `pm10`, `no2`, `so2`, `co`, `o3` (numeric pollutant features)
  - `temperature`, `humidity`, `wind_speed` (optional weather features)
  - `aqi` (numeric target label)

Example rows:

```csv
timestamp,pm2_5,pm10,no2,so2,co,o3,temperature,humidity,wind_speed,aqi
2025-01-01 00:00:00,42.1,78.4,31.2,8.0,0.72,25.3,19.5,58,2.4,118
2025-01-01 01:00:00,39.8,74.2,29.9,7.4,0.69,24.1,19.0,60,2.1,111
```

## Training

Run model training from the repository root.

```bash
python train.py \
  --data data/aqi.csv \
  --target aqi \
  --model-out artifacts/model.pkl \
  --metrics-out artifacts/metrics.json \
  --test-size 0.2 \
  --random-state 42
```

## Prediction

You can run inference with either a single feature vector or a CSV input file.

### Option A: Single feature vector

```bash
python predict.py \
  --model artifacts/model.pkl \
  --features "pm2_5=35.0,pm10=60.0,no2=22.0,so2=6.0,co=0.55,o3=30.0,temperature=21.0,humidity=50.0,wind_speed=3.0"
```

### Option B: Batch prediction from a file

```bash
python predict.py \
  --model artifacts/model.pkl \
  --input data/predict_input.csv \
  --output artifacts/predictions.csv
```

## Output Artifacts

By default, training/inference outputs should be written under `artifacts/`:

- `artifacts/model.pkl` — serialized trained model
- `artifacts/metrics.json` — evaluation metrics report (e.g., MAE/RMSE/R²)
- `artifacts/predictions.csv` — batch inference output (if using file input)
- `artifacts/plots/` — optional plots (feature importance, residuals, trend diagnostics)

## Project Structure

Key modules and directories:

- `train.py` — training entrypoint and model fitting workflow
- `predict.py` — prediction entrypoint for single/batch inference
- `data/` — raw and prepared input datasets (for example, `data/aqi.csv`)
- `artifacts/` — saved models, metrics, predictions, and optional plots
- `requirements.txt` — Python dependencies
- `README.md` — usage and operational documentation

## Limitations

- Predictions are only as reliable as the coverage and quality of historical data.
- Missing pollutant values or inconsistent sensor calibrations can significantly reduce accuracy.
- Model performance may degrade when deployed to locations/seasons not represented in training data.
- AQI definitions can vary by region; ensure your target labels use a consistent standard.
- Extreme events (wildfires, dust storms, abrupt industrial changes) may be underrepresented and harder to predict.
- Time-dependent leakage can occur if train/test splits do not respect chronology.
