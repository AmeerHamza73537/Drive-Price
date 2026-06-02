This file documents the main imports and components used throughout the project.

- pandas: data loading and DataFrame manipulations.
- numpy: numerical operations and array handling.
- joblib: persistence for scikit-learn objects (models, preprocessors).
- sklearn.impute.SimpleImputer: used to fill missing values for numeric and categorical features.
- sklearn.preprocessing.StandardScaler: used to scale numeric features to zero mean/unit variance.
- sklearn.preprocessing.OneHotEncoder: used to encode categorical features into binary indicators.
- sklearn.compose.ColumnTransformer: composes different preprocessing pipelines for numeric and categorical columns.
- sklearn.pipeline.Pipeline: helps chain imputation and scaling/encoding steps.
- sklearn.model_selection.train_test_split: splits data into training and testing subsets.
- sklearn.linear_model.LinearRegression: baseline linear model for regression.
- sklearn.ensemble.RandomForestRegressor: tree ensemble model, good baseline for tabular data.
- sklearn.ensemble.GradientBoostingRegressor: boosting-based regressor, often strong performance.
- sklearn.ensemble.AdaBoostRegressor: boosting ensemble using simple base estimators.
- sklearn.metrics: MAE, RMSE, R2 are used to quantify model performance.
- matplotlib / seaborn: plotting libraries used to create evaluation visualizations saved to `static/plots`.
- flask: lightweight web framework used to create the UI and endpoints for prediction and download.

High-level components

- `src/preprocess.py` contains loading and preprocessor-building logic.
- `src/train_models.py` orchestrates training multiple models, evaluating, saving models, and producing plots.
- `app.py` is the Flask application that serves a small frontend for batch predictions and serves generated plots.

When editing or extending the project:
- Keep the training features consistent between training and serving: the `preprocessor.joblib` built and saved during training is loaded by the Flask app.
- If you change the feature set, re-run `src/train_models.py` to refresh `models/` and `static/plots`.
