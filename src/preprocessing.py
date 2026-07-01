"""
preprocessing.py — Data Cleaning & Preprocessing
==================================================
Handles all data cleaning steps:
  - Missing value imputation (numerical & categorical)
  - Outlier detection and capping
  - Duplicate removal
  - Type corrections

Implements a sklearn-compatible HousePricePreprocessor class
with fit / transform / fit_transform interface.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils import get_logger, check_dataframe

logger = get_logger("preprocessing")

# ─── Domain-Specific NA Fill Values ────────────────────────────────────────────
# Many NA values in this dataset actually mean "None" / "No such feature"
# Reference: Kaggle data description
NONE_FILL_COLS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]

ZERO_FILL_COLS = [
    "GarageYrBlt", "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
]

# Ordinal quality mappings (for later use by feature engineering)
QUALITY_MAP = {
    "Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0, "None": 0,
}

# Columns with high outlier sensitivity to cap via IQR
OUTLIER_COLS = [
    "GrLivArea", "LotArea", "TotalBsmtSF", "1stFlrSF",
    "LotFrontage", "BsmtFinSF1",
]


class HousePricePreprocessor(BaseEstimator, TransformerMixin):
    """
    Full preprocessing pipeline for the Ames Housing dataset.

    Steps
    -----
    1. Drop ID column (if present)
    2. Remove duplicate rows
    3. Domain-specific NA fills (None / 0)
    4. Numerical median imputation
    5. Categorical mode imputation
    6. Outlier capping via IQR (fitting IQR bounds on train set)
    7. Type corrections (year columns → int, MSSubClass → str)

    Parameters
    ----------
    iqr_multiplier : float
        IQR multiplier for outlier capping (default 1.5).
    outlier_cols : list[str]
        Columns to apply IQR capping on.
    """

    def __init__(self, iqr_multiplier: float = 1.5, outlier_cols: list = None):
        self.iqr_multiplier = iqr_multiplier
        self.outlier_cols = outlier_cols or OUTLIER_COLS

    def fit(self, X: pd.DataFrame, y=None):
        """Learn imputation statistics and IQR bounds from training data."""
        df = X.copy()

        # Compute median for numerical columns
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.median_fill_ = df[num_cols].median()

        # Compute mode for categorical columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.mode_fill_ = df[cat_cols].mode().iloc[0]

        # Compute IQR bounds for outlier capping
        self.iqr_bounds_ = {}
        for col in self.outlier_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                self.iqr_bounds_[col] = (
                    Q1 - self.iqr_multiplier * IQR,
                    Q3 + self.iqr_multiplier * IQR,
                )

        # Store LotFrontage neighborhood medians for smart imputation
        if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
            self.lot_frontage_medians_ = (
                df.groupby("Neighborhood")["LotFrontage"].median()
            )
        else:
            self.lot_frontage_medians_ = None

        logger.info("Preprocessor fitted.")
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Apply all cleaning steps to the dataset."""
        df = X.copy()
        logger.info(f"Preprocessing: input shape {df.shape}")

        # ── 1. Drop Id column ──────────────────────────────────────────
        if "Id" in df.columns:
            df = df.drop(columns=["Id"])

        # ── 2. Remove duplicates ───────────────────────────────────────
        n_before = len(df)
        df = df.drop_duplicates()
        if len(df) < n_before:
            logger.info(f"Removed {n_before - len(df)} duplicate rows.")

        # ── 3. Domain-specific NA fills ───────────────────────────────
        for col in NONE_FILL_COLS:
            if col in df.columns:
                df[col] = df[col].fillna("None")

        for col in ZERO_FILL_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Smart LotFrontage fill: use neighborhood median
        if "LotFrontage" in df.columns and self.lot_frontage_medians_ is not None:
            df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
                lambda x: x.fillna(
                    self.lot_frontage_medians_.get(x.name, x.median())
                )
            )

        # ── 4. Numerical median imputation ────────────────────────────
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(self.median_fill_.get(col, 0))

        # ── 5. Categorical mode imputation ────────────────────────────
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(self.mode_fill_.get(col, "None"))

        # ── 6. Outlier capping ────────────────────────────────────────
        for col, (lower, upper) in self.iqr_bounds_.items():
            if col in df.columns:
                n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                if n_outliers > 0:
                    df[col] = df[col].clip(lower=lower, upper=upper)
                    logger.info(f"Capped {n_outliers} outliers in '{col}'")

        # ── 7. Type corrections ───────────────────────────────────────
        # MSSubClass should be treated as categorical
        if "MSSubClass" in df.columns:
            df["MSSubClass"] = df["MSSubClass"].astype(str)

        # Year columns: ensure integer type
        year_cols = ["YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold"]
        for col in year_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)

        check_dataframe(df, "After preprocessing")
        logger.info(f"Preprocessing complete: output shape {df.shape}")
        return df

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
