# DrivePrice — Car Price Predictor

End-to-end project for estimating used vehicle prices using scikit-learn and a small Flask UI.

## Summary

DrivePrice trains regression models on vehicle listing data, persists preprocessing and model artifacts, and exposes a lightweight web UI to obtain single-row price predictions. The repository includes data cleaning, a reusable preprocessing pipeline, multiple regressor training (with target transformation), evaluation plots, and a minimal Flask frontend.

## Features

- Reproducible preprocessing pipeline with median imputation and scaling for numeric features
- High-cardinality categorical reduction (`CategoryReducer`) and one-hot encoding for categorical features
- Multiple regression models: `LinearRegression`, `RandomForest`, `GradientBoosting`, `AdaBoost`
- Log-target transform (`TransformedTargetRegressor`) to stabilise targets and improve numeric behaviour
- Persisted artifacts (`models/*.joblib`) for fast inference
- Lightweight Flask UI (`app.py`, `templates/`) for quick local predictions

## Quick start (Windows)

1. Clone the repository and enter the folder:

```powershell
git clone <repository-url>
cd "Car Price Predictor"
```

2. Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # PowerShell
pip install -r requirements.txt
```

3. Add your dataset `vehicles.csv` (it must contain a `price` column) to the repository root.

4. (Optional) Train models and generate evaluation artifacts:

```powershell
python -m src.train_models
```

5. Run the Flask app locally:

```powershell
python app.py
# Open http://127.0.0.1:5000/ in your browser
```

## Usage

- Use the web form on the index page to submit vehicle attributes and receive a predicted price.
- Training outputs are written to `models/` (persisted artifacts), `static/plots/` (visualizations), and `outputs/results.csv` (metrics).

## Project layout

- `app.py` — Flask app serving the prediction form and results
- `src/preprocess.py` — dataset loading, cleaning, and `build_preprocessor`
- `src/train_models.py` — training, evaluation, and model export
- `models/` — saved preprocessor and trained models (`*.joblib`)
- `static/` — style and generated plots (`static/plots/`)
- `templates/` — Jinja2 templates for the UI
- `outputs/` — evaluation CSVs such as `results.csv`
- `vehicles.csv` — expected input dataset (not tracked in the repo)

## Preprocessing details

`src/preprocess.py` builds a `ColumnTransformer` that:

- Infers numeric and categorical columns from a sample DataFrame
- Applies median imputation and `StandardScaler` to numeric columns
- For categorical columns: most-frequent imputation, optional top-N reduction via `CategoryReducer`, then one-hot encoding

This approach keeps the feature matrix compact and stable for training.

## Training & models

`src/train_models.py` trains multiple regressors wrapped with `TransformedTargetRegressor` using `np.log1p` / `np.expm1` for the target. This reduces the influence of price skew and makes metrics more stable. Trained models and the fitted preprocessor are saved to `models/` as `.joblib` files.

## Running in production

- The Flask app loads any `.joblib` models found in `models/` at startup.
- For production, serve the Flask app with a WSGI server (e.g. Gunicorn) behind a reverse proxy, enable logging and HTTPS, and secure model artifact access.

## Dependencies

Core dependencies (see `requirements.txt`): `Flask`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `numpy`.

## Notes

- Ensure `vehicles.csv` contains a `price` column before training.
- If you rename or reorder fields exposed in the UI, update `DEFAULT_FEATURE_COLUMNS` in `src/preprocess.py` and the form in `templates/index.html`.
- Use `validate_model.py` to quickly sanity-check saved preprocessor/model pairs.

## Contributing

Contributions are welcome. Please open focused pull requests and run the validation scripts locally before submitting. If you want help adding CI, tests, or a `CONTRIBUTING.md`, tell me which services to target.

## License

Add your preferred license text here.

---

If you want the README adjusted (additional badges, a short demo GIF, or a CONTRIBUTING guide), tell me what you'd like and I'll add it.
