"""
Training and evaluation script for multiple regression models.
This script reads the dataset, uses the preprocessor from `preprocess.py`,
trains several regressors, evaluates them on a hold-out test set, saves
models and plots to the `models/` and `static/plots/` folders, and records
metrics to `outputs/results.csv`.
"""

# Data handling
import pandas as pd
import numpy as np
import os
import joblib

# Models and evaluation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Local preprocessing utilities
from src.preprocess import build_preprocessor, load_data, clean_data, DEFAULT_FEATURE_COLUMNS


# Ensure output directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("static/plots", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


def train_and_evaluate(csv_path: str, target_column: str = "price", test_size: float = 0.2, random_state: int = 42, nrows: int = None):
    """Train multiple regressors and record evaluation results.

    Args:
        csv_path: Path to dataset CSV file.
        target_column: Name of the target column.
        test_size: Fraction of dataset to reserve for testing.
        random_state: RNG seed for reproducibility.
        nrows: Optional number of rows to read (for quick runs).

    Returns:
        results_df: DataFrame with evaluation metrics for each model.
    """
    # Load data
    df = load_data(csv_path, nrows=nrows)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe columns: {df.columns.tolist()}")

    # Clean bad target values and extreme outliers
    df = clean_data(df, target_column=target_column)

    # Build preprocessor using the fields available in the UI
    preprocessor, feature_names = build_preprocessor(
        df,
        target_column=target_column,
        feature_columns=DEFAULT_FEATURE_COLUMNS,
    )

    # Split around the cleaned features only
    X = df[DEFAULT_FEATURE_COLUMNS]
    y = df[target_column].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Fit preprocessor on training set and transform
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Save preprocessor
    joblib.dump(preprocessor, "models/preprocessor.joblib")

        # Define models to train. We wrap each regressor in a
        # `TransformedTargetRegressor` (below) to apply a stable log transform to
        # the target during training and automatically invert the transform at
        # prediction time. This helps when the target (price) is skewed.
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
            "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
            "AdaBoost": AdaBoostRegressor(random_state=random_state),
        }

    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        # Wrap the base regressor to transform the target using log1p during
        # fitting and expm1 when producing predictions. This reduces the
        # influence of extreme high-price outliers and stabilizes training.
        wrapped_model = TransformedTargetRegressor(
            regressor=model,
            func=np.log1p,
            inverse_func=np.expm1,
        )
        wrapped_model.fit(X_train_trans, y_train)
        preds = wrapped_model.predict(X_test_trans)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2})

        # Save model
        model_path = f"models/{name}.joblib"
        joblib.dump(wrapped_model, model_path)

        # Scatter plot: actual vs predicted
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=y_test, y=preds, alpha=0.4)
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title(f"Actual vs Predicted - {name}")
        lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
        plt.plot(lims, lims, "k--", alpha=0.7)
        plt.tight_layout()
        plot_path = f"static/plots/{name}_pred_vs_actual.png"
        plt.savefig(plot_path)
        plt.close()

    # Save results table
    results_df = pd.DataFrame(results).sort_values("rmse")
    results_df.to_csv("outputs/results.csv", index=False)

    # Plot metrics bar chart
    plt.figure(figsize=(8, 4))
    sns.barplot(data=results_df.melt(id_vars=["model"] , value_vars=["rmse", "mae", "r2"], var_name="metric", value_name="value"),
                x="model", y="value", hue="metric")
    plt.title("Model evaluation metrics")
    plt.tight_layout()
    plt.savefig("static/plots/model_metrics.png")
    plt.close()

    return results_df


if __name__ == "__main__":
    # When executed directly, train on the full CSV in the workspace root
    csv_path = "vehicles.csv"
    print("Starting training process. This may take several minutes depending on dataset size and model complexity.")
    res = train_and_evaluate(csv_path)
    print("Training complete. Results:\n", res)
