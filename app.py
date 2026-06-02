"""
Flask application that provides a simple UI for the Car Price Predictor.

Purpose:
- Serve a simple web form to collect vehicle details from a user.
- Load a persisted preprocessor and ML model to produce a single price
    prediction for the submitted vehicle.
- Render evaluation plots generated during training so collaborators can
    inspect model performance.

How it works:
- The app reads available model artifacts from the `models/` directory.
- A preprocessor (if present) is applied to incoming form data so the
    model receives the same feature representation used during training.
- Predictions are returned and displayed on `results.html`.

This file contains two Flask view functions (`index` and `predict`) that are
small and documented below. See function docstrings for more details.
"""

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import os
import numpy as np

# Explanation of key imports:
# - `flask` provides the web framework and templating helpers used by the app.
# - `pandas` is used here only to probe the training CSV for column names.
# - `joblib` loads saved sklearn objects (preprocessor and models).
# - `os` is used to list and compose filesystem paths.
# - `numpy` is used for numeric placeholders (e.g. `np.nan`).

app = Flask(__name__)

# Load available models and the preprocessor if they exist
MODEL_DIR = "models"
AVAILABLE_MODELS = {}
for fname in os.listdir(MODEL_DIR):
    if fname.endswith(".joblib") and fname != "preprocessor.joblib":
        model_name = os.path.splitext(fname)[0]
        AVAILABLE_MODELS[model_name] = os.path.join(MODEL_DIR, fname)

BEST_MODEL = "RandomForest"
if BEST_MODEL not in AVAILABLE_MODELS:
    BEST_MODEL = next(iter(AVAILABLE_MODELS.keys()), None)

PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")
PREPROCESSOR = None
if os.path.exists(PREPROCESSOR_PATH):
    PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)

# Define the input fields for the form
INPUT_FIELDS = ["year", "manufacturer", "model", "condition", "cylinders", "fuel", "transmission_type", "paint_color", "description"]

# Load training data to infer expected columns and provide defaults
TRAINING_COLUMNS = []
try:
    df_sample = pd.read_csv("vehicles.csv", nrows=1, low_memory=False)
    TRAINING_COLUMNS = df_sample.columns.tolist()
except Exception as e:
    print(f"Warning: Could not load training columns: {e}")


@app.route("/")
def index():
    """Render the main prediction form and show available evaluation plots.

    The function collects PNGs found in `static/plots` and passes them to
    the `index.html` template along with the `best_model` selection and the
    list of input fields expected by the UI.
    """
    plots = []
    plots_dir = os.path.join("static", "plots")
    if os.path.isdir(plots_dir):
        plots = [f"plots/{p}" for p in os.listdir(plots_dir) if p.endswith('.png')]
    return render_template("index.html", best_model=BEST_MODEL, plots=plots, fields=INPUT_FIELDS)


@app.route("/predict", methods=["POST"])
def predict():
    """Handle the form submission and return a single predicted price.

    Steps:
    1. Determine which model to use (hidden `ml_model` form field or `BEST_MODEL`).
    2. Build a single-row pandas DataFrame with columns matching the
       preprocessor's `feature_names_in_` if available.
    3. Apply the preprocessor to obtain the numeric feature array `X`.
    4. Load the selected model and call `predict(X)`.
    5. Format the numeric prediction as currency and render `results.html`.

    Returns an HTTP error (400/404) when required inputs are invalid or the
    selected model artifact cannot be found.
    """
    model_name = request.form.get("ml_model") or BEST_MODEL
    if model_name not in AVAILABLE_MODELS:
        return "Model not found", 404

    # Extract form data and keep raw values for display
    form_data = {}
    for field in INPUT_FIELDS:
        form_data[field] = request.form.get(field, "")

    # Map form inputs to the exact columns expected by the fitted preprocessor
    if PREPROCESSOR is not None and hasattr(PREPROCESSOR, "feature_names_in_"):
        expected_columns = list(PREPROCESSOR.feature_names_in_)
    else:
        expected_columns = [col for col in TRAINING_COLUMNS if col in ["year", "manufacturer", "model", "condition", "cylinders", "fuel", "transmission", "paint_color", "description"]]

    # Start with a row of NaN values for all expected columns
    full_row = {col: np.nan for col in expected_columns}

    # Numeric year is used directly; missing values stay NaN
    try:
        full_row["year"] = int(form_data["year"]) if form_data["year"] else np.nan
    except ValueError:
        return "Error: year must be numeric", 400

    # Cylinders is stored in training data as a categorical value like '4 cylinders'
    if form_data["cylinders"]:
        try:
            cylinder_count = int(form_data["cylinders"])
            full_row["cylinders"] = f"{cylinder_count} cylinders"
        except ValueError:
            full_row["cylinders"] = form_data["cylinders"]

    # Transmission type should map to the training feature name 'transmission'
    if form_data["transmission_type"] and "transmission" in expected_columns:
        full_row["transmission"] = form_data["transmission_type"]

    # Populate categorical text fields only if they are expected by the preprocessor
    for field in ["manufacturer", "model", "condition", "fuel", "paint_color", "description"]:
        if field in expected_columns:
            full_row[field] = form_data[field] or np.nan

    df = pd.DataFrame([full_row], columns=expected_columns)

    # Load selected model
    model = joblib.load(AVAILABLE_MODELS[model_name])

    # Apply preprocessor if present
    if PREPROCESSOR is not None:
        X = PREPROCESSOR.transform(df)
    else:
        X = df.values

    pred = model.predict(X)[0]

    # Format prediction result
    prediction_text = f"${pred:,.2f}"

    # Show results page with prediction
    return render_template("results.html", prediction=prediction_text, model_name=model_name, form_data=form_data)


if __name__ == '__main__':
    app.run(debug=True)

