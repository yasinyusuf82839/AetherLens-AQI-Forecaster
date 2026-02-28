# AetherLens-AQI-Forecaster
See tomorrow’s air quality today—Python + machine learning that predicts AQI from pollutant signals.

## Baseline training and evaluation
`src/models/train.py` now performs a deterministic train/validation split, trains a baseline `RandomForestRegressor`, evaluates on the validation split, and emits reproducible artifacts.

### Run
```bash
python src/models/train.py --data-path data/processed/aqi_training.csv --target AQI
```

### Validation metrics
The script computes these metrics on the validation split and both prints and saves them:
- **MAE (Mean Absolute Error):** average absolute prediction error in AQI units. Lower is better.
- **RMSE (Root Mean Squared Error):** penalizes larger errors more strongly than MAE. Lower is better.
- **R² (Coefficient of Determination):** fraction of target variance explained by the model. Closer to 1 is better.

Interpret this baseline by comparing future models against these values:
- Lower MAE and RMSE than baseline indicate improved absolute and large-error performance.
- Higher R² than baseline indicates improved explanatory/predictive signal capture.

### Artifacts produced
Artifacts are written to the `artifacts/` directory by default:
- `artifacts/metrics.json`: validation MAE, RMSE, R², and run configuration (seed/split/target/data path).
- `artifacts/feature_importance.csv`: sorted model feature importances for quick baseline interpretability.

### Determinism
A fixed random seed (`--seed`, default `42`) is applied to Python, NumPy, train/validation split, and model initialization to make baseline runs reproducible.
