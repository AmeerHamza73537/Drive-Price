"""
Preprocessing utilities for the Car Price Predictor project.

This module contains functions to load the dataset, perform cleaning,
feature engineering, and build a scikit-learn ColumnTransformer that can be
used in training and inference pipelines.

Key components:
- `clean_data`: remove invalid target rows and extreme price outliers.
- `CategoryReducer`: custom transformer that reduces high-cardinality
    categorical features to top-N categories and maps others to `__OTHER__`.
- `build_preprocessor`: compose numeric and categorical pipelines into a
    `ColumnTransformer` suitable for fitting and later transforming DataFrames.
"""

# pandas is used for data loading and DataFrame operations
import pandas as pd
# numpy is used for numerical operations
import numpy as np
# joblib is used to save/load fitted transformers (used by preprocess_and_save)
import joblib
# sklearn utilities for preprocessing and pipeline building
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

DEFAULT_FEATURE_COLUMNS = [
    "year",
    "manufacturer",
    "model",
    "condition",
    "cylinders",
    "fuel",
    "transmission",
    "paint_color",
    "description",
]


def clean_data(df: pd.DataFrame, target_column: str = "price", max_price_quantile: float = 0.999):
    """Clean the dataset before fitting the preprocessor or training models.

    This drops missing or non-positive target values and removes extreme price
    outliers that are not representative of typical vehicle listings.
    """
    df = df.copy()
    df = df.dropna(subset=[target_column])
    df = df[df[target_column] > 0]
    if 0 < max_price_quantile < 1.0:
        max_price = df[target_column].quantile(max_price_quantile)
        df = df[df[target_column] <= max_price]
    return df


class CategoryReducer(BaseEstimator, TransformerMixin):
    """Reduce high-cardinality categorical features to top categories.

    The transformer keeps only the top-N most frequent categories for each input
    column and maps all other values to the token __OTHER__. This avoids large
    one-hot encoded matrices from extremely high-cardinality columns.
    """
    def __init__(self, top_n=50):
        self.top_n = top_n
        self.top_categories_ = {}

    def fit(self, X, y=None):
        # Accept either a numpy array or DataFrame as input (compatibility with
        # sklearn pipelines which may provide arrays). Store top categories per
        # column index in `self.top_categories_` for use in `transform`.
        if isinstance(X, np.ndarray):
            Xdf = pd.DataFrame(X)
        else:
            Xdf = X.copy()

        for i, _ in enumerate(Xdf.columns):
            vals = Xdf.iloc[:, i].astype(str)
            top = vals.value_counts().nlargest(self.top_n).index.tolist()
            self.top_categories_[i] = set(top)
        return self

    def transform(self, X):
        # Convert arrays to DataFrame for easier column-wise operations. Map
        # any value not in the previously-computed top set to the token
        # '__OTHER__' to limit the downstream one-hot encoding cardinality.
        if isinstance(X, np.ndarray):
            Xdf = pd.DataFrame(X)
        else:
            Xdf = X.copy()

        Xout = Xdf.copy()
        for i in range(Xdf.shape[1]):
            top_set = self.top_categories_.get(i, set())
            col_series = Xdf.iloc[:, i].astype(str)
            Xout.iloc[:, i] = col_series.where(col_series.isin(top_set), other="__OTHER__")
        # Return numpy array to remain compatible with sklearn transformers
        return Xout.values


def load_data(csv_path: str, nrows: int = None):
    """Load CSV into a DataFrame.

    Args:
        csv_path: Path to the CSV file.
        nrows: Optional number of rows to read (useful for quick tests).

    Returns:
        pandas.DataFrame with the loaded data.
    """
    df = pd.read_csv(csv_path, nrows=nrows, low_memory=False)
    return df


def build_preprocessor(df: pd.DataFrame, target_column: str = "price", feature_columns: list[str] | None = None):
    """Build a ColumnTransformer to preprocess numeric and categorical features.

    The function:
    - uses only explicit feature columns when provided
    - drops the target column if present
    - creates pipelines for numeric and categorical processing

    Args:
        df: Example DataFrame used to infer columns.
        target_column: Name of the target column in the dataset.
        feature_columns: Optional list of columns to keep for preprocessing.

    Returns:
        preprocessor: sklearn ColumnTransformer
        feature_names: list of output feature names (approximate, for reference)
    """
    if feature_columns is None:
        cols = df.columns.tolist()
    else:
        cols = [c for c in feature_columns if c in df.columns]

    if target_column in cols:
        cols = [c for c in cols if c != target_column]

    # Infer numeric and categorical columns using pandas dtypes
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in cols if c not in numeric_cols]

    # Numeric pipeline: impute missing values then scale
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # When categorical features have very high cardinality (many unique values),
    # one-hot encoding can explode memory usage. To avoid that, we split
    # categorical columns into low-cardinality and high-cardinality groups.
    # - low-cardinality: OneHotEncode normally
    # - high-cardinality: reduce categories to the top-N most frequent and map
    #   the rest to a special "__OTHER__" token, then OneHotEncode

    CARDINALITY_THRESHOLD = 50
    unique_counts = {c: int(df[c].nunique(dropna=True)) for c in categorical_cols}
    low_cardinality = [c for c in categorical_cols if unique_counts.get(c, 0) <= CARDINALITY_THRESHOLD]
    high_cardinality = [c for c in categorical_cols if unique_counts.get(c, 0) > CARDINALITY_THRESHOLD]

    try:
        onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))

    if low_cardinality:
        low_cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", onehot_encoder),
        ])
        transformers.append(("low_cat", low_cat_pipeline, low_cardinality))

    if high_cardinality:
        high_cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("reducer", CategoryReducer(top_n=CARDINALITY_THRESHOLD)),
            ("onehot", onehot_encoder),
        ])
        transformers.append(("high_cat", high_cat_pipeline, high_cardinality))

    preprocessor = ColumnTransformer(transformers)

    feature_names = numeric_cols + [f"cat_{c}" for c in categorical_cols]

    return preprocessor, feature_names


def preprocess_and_save(csv_path: str, preprocessor_path: str = "models/preprocessor.joblib",
                        target_column: str = "price", nrows: int = None):
    """Load data, fit preprocessor on it, and save the preprocessor for reuse.

    Args:
        csv_path: Path to the CSV file.
        preprocessor_path: Where to save the fitted preprocessor.
        target_column: Target column name.
        nrows: Optional number of rows to read.

    Returns:
        df: Loaded DataFrame
        preprocessor: Fitted preprocessor object
    """
    df = load_data(csv_path, nrows=nrows)
    df = clean_data(df, target_column=target_column)
    preprocessor, feature_names = build_preprocessor(
        df,
        target_column=target_column,
        feature_columns=DEFAULT_FEATURE_COLUMNS,
    )

    # Fit the preprocessor on the cleaned dataset
    X = df[DEFAULT_FEATURE_COLUMNS]
    preprocessor.fit(X)

    # Persist the fitted preprocessor
    joblib.dump(preprocessor, preprocessor_path)

    return df, preprocessor
