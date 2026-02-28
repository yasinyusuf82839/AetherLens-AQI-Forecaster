# Dataset Expectations

Place source CSV files for training/inference in this directory.

## Required columns

A baseline AQI training file is expected to include the following columns:

- `PM2.5`: Fine particulate matter concentration
- `PM10`: Coarse particulate matter concentration
- `NO2`: Nitrogen dioxide concentration
- `SO2`: Sulfur dioxide concentration
- `CO`: Carbon monoxide concentration
- `O3`: Ozone concentration
- `AQI`: Air Quality Index target label

## Example row

```csv
PM2.5,PM10,NO2,SO2,CO,O3,AQI
35.0,68.0,21.3,8.2,0.7,39.5,92
```

## Workflow

Train a baseline model:

```bash
python -m src.models.train --data-path data/your_file.csv
```

Predict AQI from feature inputs:

```bash
python -m src.models.predict \
  --model-path artifacts/random_forest_aqi.joblib \
  --features-json '{"PM2.5": 35, "PM10": 68, "NO2": 21.3, "SO2": 8.2, "CO": 0.7, "O3": 39.5}'
```
