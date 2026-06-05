Project: Car Price Predictor

Purpose
- Predict used car prices using a tabular regression pipeline.
- Provide a model training workflow, evaluation metrics, and a Flask UI for prediction and batch data output.

Data and Inputs
- Primary dataset: `vehicles.csv`.
- Feature preprocessing: numerical imputation + scaling, categorical imputation + one-hot encoding.
- The same preprocessing pipeline is used for training and prediction via `models/preprocessor.joblib`.

Modeling and Evaluation
- Trained models:
  - `LinearRegression`
  - `RandomForestRegressor`
  - `GradientBoostingRegressor`
  - `AdaBoostRegressor`
- Evaluation metrics produced during training:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R2 score
- Best-performing model on the current test split: `RandomForest`
  - RMSE: 8,346.12
  - MAE: 4,581.46
  - R2: 0.6725
- Other model performance on the current dataset:
  - `GradientBoosting`: RMSE 10,919.76, MAE 6,867.24, R2 0.4394
  - `LinearRegression`: RMSE 12,347.54, MAE 8,147.86, R2 0.2832
  - `AdaBoost`: RMSE 18,411.32, MAE 13,115.64, R2 -0.5937
- Model artifacts saved under `models/`:
  - `LinearRegression.joblib`
  - `RandomForest.joblib`
  - `GradientBoosting.joblib`
  - `AdaBoost.joblib`
  - `preprocessor.joblib`

Core Components
- `src/preprocess.py`
  - Loads `vehicles.csv`.
  - Builds the preprocessing pipeline.
  - Encodes categorical variables and scales numeric variables.

- `src/train_models.py`
  - Trains all candidate models.
  - Evaluates each model and produces performance summaries.
  - Saves trained models and the preprocessing pipeline.
  - Generates plots and output data for reporting.

- `app.py`
  - Flask application with endpoints for predictions and results.
  - Uses the saved `preprocessor.joblib` and selected model artifact.
  - Supports batch predictions and download of results.

Outputs for Reports and Presentations
- `outputs/results.csv` contains model comparison metrics that can populate tables and charts.
- `static/plots/` contains evaluation visualizations generated during training:
  - `model_metrics.png`: grouped bar chart comparing RMSE, MAE, and R2 across all models.
  - `RandomForest_pred_vs_actual.png`: actual vs predicted scatter plot for the best performing model.
  - `GradientBoosting_pred_vs_actual.png`: actual vs predicted scatter plot for the second-best model.
  - `LinearRegression_pred_vs_actual.png`: baseline regression prediction scatter plot.
  - `AdaBoost_pred_vs_actual.png`: prediction scatter plot for the weakest model.
- `README.md` and this `COMMENTS.md` summarize project purpose, components, and usage.

Graph and Data Summary for Presentations
- Performance ranking by RMSE: `RandomForest` > `GradientBoosting` > `LinearRegression` > `AdaBoost`.
- `RandomForest` is the recommended production model based on current test split.
- The `model_metrics.png` chart is ideal for a single slide that compares all models at once.
- Each `*_pred_vs_actual.png` plot can be used to explain model calibration, bias, and variance.
- Use `outputs/results.csv` values to create data tables, annotated callouts, or trend bullets.

Suggested Report Structure
1. Project overview and business goal
2. Dataset and preprocessing strategy
3. Model candidates and training workflow
4. Evaluation metrics and best model selection
5. Visual evidence from plots and results tables
6. Deployment strategy with `app.py` and model artifacts
7. Recommendations and next steps

Detailed slide/content ideas
- Data slide: explain `vehicles.csv`, features, missing value handling, and preprocessing pipeline.
- Modeling slide: list the four models and describe the use of log-target stabilization via `TransformedTargetRegressor`.
- Evaluation slide: show RMSE/MAE/R2 table and highlight `RandomForest` as best performer.
- Graph slide: include `model_metrics.png` with a note that higher R2 and lower RMSE/MAE are better.
- Fit-quality slide: show one or two actual vs predicted plots, preferably `RandomForest_pred_vs_actual.png`.
- Deployment slide: explain how `models/preprocessor.joblib` and `models/RandomForest.joblib` are loaded in `app.py`.
- Output slide: mention `outputs/results.csv` as the report-ready export used for batch prediction summaries.

Extensions and Future Work
- Add a `validation` dataset or cross-validation report for more robust performance estimates.
- Include feature importance or SHAP values to explain model decisions.
- Add a `results_report.pdf` generator or dashboard for stakeholder-friendly exports.
- Expand the Flask UI to let users select the model and view performance summaries directly.

Extension Notes
- If the feature set changes, re-run `src/train_models.py` to refresh models and plots.
- Keep `preprocessor.joblib` synchronized with the model artifacts used in `app.py`.
- Update `README.md` and this file when new models, metrics, or outputs are added.
