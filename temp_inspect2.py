"""
Another quick dataset inspection helper that prints overall row counts and
price quantiles. Useful for identifying extreme outliers in the `price` column.
"""

import pandas as pd

# Load the full CSV (can be slow for very large files) — used for quick ad-hoc checks
sample = pd.read_csv('vehicles.csv', low_memory=False)

print('rows', len(sample))
print('price non-null', sample['price'].notna().sum())
print('price unique', sample['price'].nunique())

# Print several quantiles to inspect tail behavior (outliers)
print(sample['price'].quantile([0.001,0.01,0.05,0.25,0.5,0.75,0.9,0.95,0.99,0.999]).to_string())

print('non-null feature rates:')
for col in ['year','manufacturer','model','condition','cylinders','fuel','transmission','paint_color','description','county']:
    print(col, sample[col].notna().mean())
