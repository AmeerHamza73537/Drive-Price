"""
Small validation helper script used during development to sanity-check that
the saved preprocessor and a trained model produce reasonable outputs on a
hand-crafted set of examples.

This script is intentionally simple and prints results to stdout; it is not
part of the Flask application. Run it directly with `python validate_model.py`.
"""

import pandas as pd
import joblib
import numpy as np

# Load the preprocessor saved during training. This object transforms raw
# DataFrame rows into the numeric feature arrays the models expect.
pre = joblib.load("models/preprocessor.joblib")
print("Preprocessor loaded. Feature names:", len(pre.feature_names_in_) if hasattr(pre, "feature_names_in_") else "unknown")


def make_test_input(year, manufacturer, model_name, condition, cylinders, fuel, transmission, paint_color, description):
    """Create a one-row DataFrame matching the expected columns used by the preprocessor.

    This helper avoids repeating the dictionary literal each time.
    """
    return pd.DataFrame([{
        "year": year,
        "manufacturer": manufacturer,
        "model": model_name,
        "condition": condition,
        "cylinders": cylinders,
        "fuel": fuel,
        "transmission": transmission,
        "paint_color": paint_color,
        "description": description
    }])


model = joblib.load("models/RandomForest.joblib")
print("Model loaded: RandomForest")

# Example 1: 2015 Toyota Camry
test_input = make_test_input(2015, "Toyota", "Camry", "good", "4 cylinders", "gas", "automatic", "white", "Nice car in good condition")
X_test = pre.transform(test_input)
print("Transformed features shape:", X_test.shape)
pred = model.predict(X_test)[0]
print(f"Prediction for 2015 Toyota Camry (good condition): ${pred:,.2f}")

# Example 2: 2020 Honda Accord
test2 = make_test_input(2020, "Honda", "Accord", "excellent", "4 cylinders", "gas", "automatic", "black", "Excellent condition, low mileage")
pred2 = model.predict(pre.transform(test2))[0]
print(f"Prediction for 2020 Honda Accord (excellent): ${pred2:,.2f}")

# Example 3: 2010 Ford F-150
test3 = make_test_input(2010, "Ford", "F-150", "fair", "8 cylinders", "gas", "automatic", "silver", "Used truck, needs some work")
pred3 = model.predict(pre.transform(test3))[0]
print(f"Prediction for 2010 Ford F-150 (fair): ${pred3:,.2f}")
