"""
Quick inspection helper for the dataset. This small script loads a sample
of the CSV and prints column coverage statistics and a few sample rows.

Use it interactively while exploring `vehicles.csv` during preprocessing
and feature engineering. It is not required for the main training or app.
"""

import pandas as pd

# Read a 2000-row sample to get quick statistics without loading the full file
sample = pd.read_csv("vehicles.csv", nrows=2000, low_memory=False)

# Print column list and basic shape information
print("COLUMNS", list(sample.columns))
print("SHAPE", sample.shape)

# Print the density (fraction non-null) of top columns to understand missingness
print("DENSITY", sample.notna().mean().sort_values(ascending=False).head(30).to_string())

# Show basic statistics for the price column
print("PRICE_STATS", sample["price"].describe().to_string())

# Show a few example rows focused on the fields we care about
print("SAMPLES")
print(sample[["price","year","manufacturer","model","condition","cylinders","fuel","transmission","paint_color","description","region","state"]].head(10).to_string())
