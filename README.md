
# Infectious Diseases Prediction (Temporal + Region Validation)

Pipeline to predict **next-year high incidence** of infectious diseases at county level using XGBoost, with:
- **Temporal split** (train on past years, test on future years)
- **Region-wise evaluation** (per-county metrics on temporal test)
- **No leakage**: excludes outcome-like columns and predicts the *next* year's outcome

## Dataset (link only)
Kaggle: *Infectious Diseases, County, Year, and Sex*  
URL: https://www.kaggle.com/datasets/zayanmakar/infectious-diseases-county-year-and-sex

> Place the CSV in the project root as:
```
infectious-diseases-by-county-year-and-sex.csv
```

## Quickstart
```bash
pip install -r requirements.txt
python src/pipeline.py
```

## Settings
- `TARGET_QUANTILE = 0.85` defines "high incidence next year"
- `PRED_THRESHOLD = 0.80` used for region-wise evaluation
- `N_ESTIMATORS = 500`, `MAX_DEPTH = 4`, `ETA = 0.08`

Outputs:
- `*_roc.png`, `*_beeswarm.png`
- `*_metrics_by_threshold.csv/.json`
- `*_region_eval_thrXX.csv`
