"""
feature_engineering.py — Feature Engineering Pipeline
=======================================================
Transforms cleaned data into a model-ready feature matrix:
  1. New feature creation (domain-driven)
  2. Skewness correction via log1p / Box-Cox
  3. Categorical encoding (ordinal + one-hot)
  4. Near-zero variance and high-correlation feature removal
  5. Robust scaling of numerical features

Implements a sklearn-compatible FeatureEngineer class.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler

from src.utils import get_logger

logger = get_logger("feature_engineering")

# ─── Ordinal Encoding Maps ──────────────────────────────────────────────────────
ORDINAL_MAPS = {
    "ExterQual":   {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "None": 0, "NA": 0},
    "ExterCond":   {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "None": 0, "NA": 0},
    "BsmtQual":    {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtCond":    {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtExposure":{"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1":{"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2":{"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "HeatingQC":   {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "None": 0},
    "KitchenQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "None": 0},
    "FireplaceQu": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageQual":  {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageCond":  {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageFinish":{"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "LandSlope":   {"Gtl": 1, "Mod": 2, "Sev": 3},
    "LotShape":    {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
    "PavedDrive":  {"N": 0, "P": 1, "Y": 2},
    "PoolQC":      {"None": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "Fence":       {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
    "Functional":  {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8},
    "Utilities":   {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4},
}

# Skewness threshold — features above this get log1p-transformed
SKEW_THRESHOLD = 0.75


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Full feature engineering pipeline for the Ames Housing dataset.

    Steps
    -----
    1. Create new derived features
    2. Apply ordinal encoding to quality/condition columns
    3. One-hot encode remaining categorical columns
    4. Fix skewed numerical features with log1p transformation
    5. Drop near-zero-variance features (std < 0.01)
    6. Drop highly correlated features (Pearson r > 0.95)
    7. Scale all numerical features with RobustScaler

    Parameters
    ----------
    skew_threshold : float   Minimum skewness for log1p transform (default 0.75)
    corr_threshold : float   Max allowed correlation between features (default 0.95)
    scale          : bool    Whether to apply RobustScaler (default True)
    """

    def __init__(
        self,
        skew_threshold: float = SKEW_THRESHOLD,
        corr_threshold: float = 0.95,
        scale: bool = True,
    ):
        self.skew_threshold = skew_threshold
        self.corr_threshold = corr_threshold
        self.scale = scale

    # ── Private helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _create_new_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add domain-driven engineered features."""
        current_year = 2010  # Approximate dataset collection year

        # Total living area (key predictor)
        df["TotalSF"] = (
            df.get("TotalBsmtSF", 0)
            + df.get("1stFlrSF", 0)
            + df.get("2ndFlrSF", 0)
        )

        # Total bathrooms (weighted: half-baths count as 0.5)
        df["TotalBaths"] = (
            df.get("FullBath", 0)
            + 0.5 * df.get("HalfBath", 0)
            + df.get("BsmtFullBath", 0)
            + 0.5 * df.get("BsmtHalfBath", 0)
        )

        # Age features
        if "YearBuilt" in df.columns:
            df["HouseAge"] = current_year - df["YearBuilt"]
        if "YearRemodAdd" in df.columns:
            df["RemodAge"] = current_year - df["YearRemodAdd"]
            if "YearBuilt" in df.columns:
                df["IsRemodeled"] = (df["YearRemodAdd"] > df["YearBuilt"]).astype(int)
        if "GarageYrBlt" in df.columns:
            df["GarageAge"] = current_year - df["GarageYrBlt"]

        # Binary presence indicators
        df["HasPool"]      = (df.get("PoolArea", 0) > 0).astype(int)
        df["HasFireplace"] = (df.get("Fireplaces", 0) > 0).astype(int)
        df["HasGarage"]    = (df.get("GarageArea", 0) > 0).astype(int)
        df["Has2ndFloor"]  = (df.get("2ndFlrSF", 0) > 0).astype(int)
        df["HasBsmt"]      = (df.get("TotalBsmtSF", 0) > 0).astype(int)
        df["HasMasVnr"]    = (df.get("MasVnrArea", 0) > 0).astype(int)
        df["HasWoodDeck"]  = (df.get("WoodDeckSF", 0) > 0).astype(int)
        df["HasPorch"]     = (
            df.get("OpenPorchSF", 0)
            + df.get("EnclosedPorch", 0)
            + df.get("3SsnPorch", 0)
            + df.get("ScreenPorch", 0)
        ).gt(0).astype(int)

        # Quality × Area interaction
        if "OverallQual" in df.columns:
            df["QualArea"]    = df["OverallQual"] * df.get("GrLivArea", 0)
            df["QualTotalSF"] = df["OverallQual"] * df["TotalSF"]

        # Total porch area
        df["TotalPorchSF"] = (
            df.get("OpenPorchSF", 0)
            + df.get("EnclosedPorch", 0)
            + df.get("3SsnPorch", 0)
            + df.get("ScreenPorch", 0)
        )

        return df

    def _apply_ordinal_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace ordinal quality/condition columns with integer codes."""
        for col, mapping in ORDINAL_MAPS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0).astype(int)
        return df

    def _fix_skewness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log1p to numerical features with |skew| > threshold."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        self.skewed_cols_ = []
        for col in num_cols:
            if col in df.columns and df[col].min() >= 0:
                sk = df[col].skew()
                if abs(sk) > self.skew_threshold:
                    df[col] = np.log1p(df[col])
                    self.skewed_cols_.append(col)
        if self.skewed_cols_:
            logger.info(f"log1p applied to {len(self.skewed_cols_)} skewed features: "
                        f"{self.skewed_cols_[:5]}{'...' if len(self.skewed_cols_) > 5 else ''}")
        return df

    def _drop_low_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove near-zero-variance numerical features."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        low_var = [c for c in num_cols if df[c].std() < 0.01]
        if low_var:
            df = df.drop(columns=low_var)
            logger.info(f"Dropped {len(low_var)} low-variance features: {low_var}")
        self.dropped_low_var_ = low_var
        return df

    def _drop_high_corr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove one feature from each highly-correlated pair."""
        num_df = df.select_dtypes(include=[np.number])
        corr_matrix = num_df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [
            col for col in upper.columns
            if any(upper[col] > self.corr_threshold)
        ]
        if to_drop:
            df = df.drop(columns=to_drop, errors="ignore")
            logger.info(f"Dropped {len(to_drop)} highly-correlated features: {to_drop[:5]}")
        self.dropped_high_corr_ = to_drop
        return df

    # ── Public sklearn API ──────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the feature engineer on training data."""
        df = X.copy()

        # Create features (needed before fitting scaler)
        df = self._create_new_features(df)
        df = self._apply_ordinal_encoding(df)

        # One-hot encode
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
        self.ohe_columns_ = df.columns.tolist()

        # Skewness
        df = self._fix_skewness(df)

        # Variance and correlation filtering
        df = self._drop_low_variance(df)
        df = self._drop_high_corr(df)

        # Final column list after all filtering
        self.final_columns_ = df.columns.tolist()

        # Fit scaler
        if self.scale:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self.num_cols_to_scale_ = num_cols
            self.scaler_ = RobustScaler()
            self.scaler_.fit(df[num_cols])

        logger.info(f"FeatureEngineer fitted. Final feature count: {len(self.final_columns_)}")
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Transform a DataFrame using the fitted pipeline."""
        df = X.copy()

        # Create features
        df = self._create_new_features(df)
        df = self._apply_ordinal_encoding(df)

        # One-hot encode
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

        # Align columns to training set (add missing as 0, drop extras)
        for col in self.ohe_columns_:
            if col not in df.columns:
                df[col] = 0
        df = df[self.ohe_columns_]  # keep only training columns

        # Skewness correction (same columns as fitted)
        for col in getattr(self, "skewed_cols_", []):
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))

        # Drop same columns as during fit
        drop_cols = (
            getattr(self, "dropped_low_var_", [])
            + getattr(self, "dropped_high_corr_", [])
        )
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Scale
        if self.scale and hasattr(self, "scaler_"):
            valid_num = [c for c in self.num_cols_to_scale_ if c in df.columns]
            df[valid_num] = self.scaler_.transform(df[valid_num])

        logger.info(f"FeatureEngineer transform complete: shape {df.shape}")
        return df

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
