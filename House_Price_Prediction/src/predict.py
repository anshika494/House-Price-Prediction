"""
predict.py — Prediction & Submission Generation
=================================================
Loads the saved pipeline and model, applies them to the
Kaggle test set, and produces submission.csv.

Usage (CLI):
    python -m src.predict
    python -m src.predict --model models/house_price_model.pkl
                          --input data/test.csv
                          --output outputs/submission.csv
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from src.utils import (
    get_logger, timer,
    DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT,
)

logger = get_logger("predict")


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_pipeline(path: str | Path = None):
    """
    Load the saved prediction pipeline (preprocessor + feature engineer + model).

    Parameters
    ----------
    path : str or Path
        Path to the saved pipeline .pkl or .joblib file.
        Defaults to models/house_price_pipeline.joblib.

    Returns
    -------
    dict with keys: preprocessor, feature_engineer, model
    """
    if path is None:
        path = MODELS_DIR / "house_price_pipeline.joblib"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Pipeline not found at: {path}")

    logger.info(f"Loading pipeline from {path}")
    pipeline = joblib.load(path)
    logger.info("Pipeline loaded successfully.")
    return pipeline


def load_model(path: str | Path = None):
    """
    Load only the trained model (without preprocessor).

    Parameters
    ----------
    path : Path to model file (.pkl or .joblib)
    """
    if path is None:
        path = MODELS_DIR / "house_price_model.pkl"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model not found at: {path}")

    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            model = pickle.load(f)
    else:
        model = joblib.load(path)

    logger.info(f"Model loaded from {path}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────────────

def predict(
    df: pd.DataFrame,
    pipeline: dict,
) -> np.ndarray:
    """
    Run the full prediction pipeline on a DataFrame.

    The pipeline must have:
        pipeline['preprocessor']      → HousePricePreprocessor
        pipeline['feature_engineer']  → FeatureEngineer
        pipeline['model']             → fitted sklearn/XGB/LGB model

    The target was log1p-transformed during training, so we reverse
    that with np.expm1 here.

    Parameters
    ----------
    df       : pd.DataFrame — raw input data (test set)
    pipeline : dict         — loaded pipeline dict

    Returns
    -------
    np.ndarray of predicted SalePrice values (in original $)
    """
    with timer("Prediction pipeline"):
        X = pipeline["preprocessor"].transform(df)
        X = pipeline["feature_engineer"].transform(X)
        y_log = pipeline["model"].predict(X)
        y_pred = np.expm1(y_log)  # reverse log1p transform

    logger.info(f"Predicted {len(y_pred)} samples. "
                f"Price range: ${y_pred.min():,.0f} — ${y_pred.max():,.0f}")
    return y_pred


# ──────────────────────────────────────────────────────────────────────────────
# Submission Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_submission(
    predictions: np.ndarray,
    ids: pd.Series,
    output_path: str | Path = None,
) -> pd.DataFrame:
    """
    Generate a Kaggle-format submission CSV.

    Parameters
    ----------
    predictions : array of predicted SalePrice values
    ids         : pd.Series of Id values from the test set
    output_path : path to save CSV (defaults to outputs/submission.csv)

    Returns
    -------
    pd.DataFrame with columns [Id, SalePrice]
    """
    if output_path is None:
        output_path = OUTPUTS_DIR / "submission.csv"

    submission = pd.DataFrame({"Id": ids.values, "SalePrice": predictions})
    submission["SalePrice"] = submission["SalePrice"].clip(lower=0)
    submission.to_csv(output_path, index=False)

    logger.info(f"Submission saved → {output_path}")
    logger.info(f"Submission stats:\n{submission['SalePrice'].describe()}")
    return submission


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate house price predictions and Kaggle submission."
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default=str(MODELS_DIR / "house_price_pipeline.joblib"),
        help="Path to the saved pipeline (.joblib)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DATA_DIR / "test.csv"),
        help="Path to test CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUTS_DIR / "submission.csv"),
        help="Path for output submission CSV",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("House Price Prediction — Inference")
    logger.info("=" * 60)

    # Load test data
    test_df = pd.read_csv(args.input)
    ids = test_df["Id"]
    logger.info(f"Loaded test data: {test_df.shape}")

    # Load pipeline and predict
    pipeline = load_pipeline(args.pipeline)
    predictions = predict(test_df, pipeline)

    # Generate submission
    submission = generate_submission(predictions, ids, args.output)
    print("\nSubmission preview:")
    print(submission.head(10).to_string(index=False))
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
