"""
train.py — Model Training, Evaluation & Hyperparameter Tuning
==============================================================
Orchestrates the full ML pipeline:
  1. Load and split data
  2. Preprocess and engineer features
  3. Train 8 regression models
  4. Evaluate with R², MAE, MSE, RMSE, CV RMSE
  5. Hyperparameter tuning of the best model
  6. Save trained pipeline + model artifacts

Usage (CLI):
    python -m src.train
    python -m src.train --data data/train.csv --seed 42
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold,
    GridSearchCV, RandomizedSearchCV,
)
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    mean_squared_error,
)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("XGBoost not installed. Skipping XGBRegressor.")

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    warnings.warn("LightGBM not installed. Skipping LGBMRegressor.")

from src.preprocessing import HousePricePreprocessor
from src.feature_engineering import FeatureEngineer
from src.utils import (
    get_logger, set_seed, save_figure, log_metrics,
    format_metrics_table, timer,
    DATA_DIR, MODELS_DIR, FIGURES_DIR,
)

warnings.filterwarnings("ignore")
logger = get_logger("train")

RANDOM_STATE = 42
CV_FOLDS = 5

# ─── Plot Style ─────────────────────────────────────────────────────────────────
DARK_BG   = "#0f0f1a"
ACCENT    = "#7c5cbf"
ACCENT2   = "#e96c7d"
PALETTE   = ["#7c5cbf", "#e96c7d", "#5bc0eb", "#ffa500", "#4caf50",
             "#f06292", "#ab47bc", "#26c6da"]
TEXT_COLOR = "#e0e0e0"

def _set_dark_style():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor":   "#1a1a2e",
        "axes.labelcolor":  TEXT_COLOR,
        "axes.edgecolor":   "#3a3a5c",
        "text.color":       TEXT_COLOR,
        "xtick.color":      TEXT_COLOR,
        "ytick.color":      TEXT_COLOR,
        "grid.color":       "#2a2a4a",
        "grid.alpha":       0.5,
        "legend.facecolor": "#1a1a2e",
        "legend.edgecolor": "#3a3a5c",
        "font.family":      "DejaVu Sans",
        "font.size":        11,
    })

_set_dark_style()


# ──────────────────────────────────────────────────────────────────────────────
# Model Definitions
# ──────────────────────────────────────────────────────────────────────────────

def get_models(random_state: int = RANDOM_STATE) -> dict:
    """Return a dictionary of all models to train."""
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=1.0, random_state=random_state),
        "Lasso Regression":  Lasso(alpha=0.001, max_iter=10000, random_state=random_state),
        "Decision Tree":     DecisionTreeRegressor(max_depth=10, random_state=random_state),
        "Random Forest":     RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_leaf=2,
            n_jobs=-1, random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=random_state,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=-1, random_state=random_state,
            verbosity=0, eval_metric="rmse",
        )
    if HAS_LGB:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            num_leaves=50, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=-1, random_state=random_state,
            verbosity=-1,
        )
    return models


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test:  pd.DataFrame,
    y_test:  np.ndarray,
    cv: int = CV_FOLDS,
) -> dict:
    """
    Evaluate a fitted model on the test set and via k-fold CV.

    Returns
    -------
    dict with R2, MAE, MSE, RMSE, CV_RMSE, CV_R2
    """
    y_pred = model.predict(X_test)

    # Reverse log transform for real-unit metrics
    y_pred_orig  = np.expm1(y_pred)
    y_test_orig  = np.expm1(y_test)

    r2   = r2_score(y_test_orig, y_pred_orig)
    mae  = mean_absolute_error(y_test_orig, y_pred_orig)
    mse  = mean_squared_error(y_test_orig, y_pred_orig)
    rmse = np.sqrt(mse)

    # Cross-validation on log scale (standard ML practice for this dataset)
    kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    cv_neg_rmse = cross_val_score(
        model, X_train, y_train, cv=kf,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    cv_rmse = -cv_neg_rmse.mean()
    cv_r2   = cross_val_score(
        model, X_train, y_train, cv=kf,
        scoring="r2", n_jobs=-1,
    ).mean()

    return {
        "R2":      r2,
        "MAE":     mae,
        "MSE":     mse,
        "RMSE":    rmse,
        "CV_RMSE": cv_rmse,
        "CV_R2":   cv_r2,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Visualizations
# ──────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(results: list[dict]) -> None:
    """Bar chart comparing model R² and RMSE scores."""
    df = pd.DataFrame(results).sort_values("R2", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(DARK_BG)

    colors = PALETTE[:len(df)]

    # R² bar chart
    axes[0].barh(df["model"], df["R2"], color=colors, edgecolor="none", height=0.6)
    axes[0].set_xlabel("R² Score", fontsize=12)
    axes[0].set_title("Model R² Score Comparison", fontsize=14, fontweight="bold", pad=15)
    axes[0].axvline(0.9, color="#ffffff", linestyle="--", alpha=0.3, label="R²=0.9")
    axes[0].legend(fontsize=9)
    for i, v in enumerate(df["R2"]):
        axes[0].text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=9)

    # RMSE bar chart
    axes[1].barh(df["model"], df["RMSE"], color=colors, edgecolor="none", height=0.6)
    axes[1].set_xlabel("RMSE ($)", fontsize=12)
    axes[1].set_title("Model RMSE Comparison", fontsize=14, fontweight="bold", pad=15)
    for i, v in enumerate(df["RMSE"]):
        axes[1].text(v + 100, i, f"${v:,.0f}", va="center", fontsize=9)

    plt.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, "model_comparison")
    plt.close(fig)


def plot_actual_vs_predicted(
    model, X_test: pd.DataFrame, y_test: np.ndarray, model_name: str
) -> None:
    """Scatter plot of actual vs predicted prices."""
    y_pred = np.expm1(model.predict(X_test))
    y_true = np.expm1(y_test)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(DARK_BG)

    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color=ACCENT, edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "w--", alpha=0.6, lw=2, label="Perfect prediction")
    ax.set_xlabel("Actual Sale Price ($)", fontsize=12)
    ax.set_ylabel("Predicted Sale Price ($)", fontsize=12)
    ax.set_title(f"Actual vs. Predicted — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    save_figure(fig, f"actual_vs_predicted_{model_name.replace(' ', '_').lower()}")
    plt.close(fig)


def plot_residuals(
    model, X_test: pd.DataFrame, y_test: np.ndarray, model_name: str
) -> None:
    """Residual plot: predicted vs residuals."""
    y_pred = np.expm1(model.predict(X_test))
    y_true = np.expm1(y_test)
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(DARK_BG)

    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=20, color=ACCENT2, edgecolors="none")
    axes[0].axhline(0, color="white", linestyle="--", alpha=0.6, lw=2)
    axes[0].set_xlabel("Predicted Price ($)", fontsize=12)
    axes[0].set_ylabel("Residuals ($)", fontsize=12)
    axes[0].set_title("Residuals vs. Predicted", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.2)

    # Residual distribution
    axes[1].hist(residuals, bins=50, color=ACCENT, edgecolor="none", alpha=0.85)
    axes[1].axvline(0, color="white", linestyle="--", alpha=0.7, lw=2)
    axes[1].set_xlabel("Residuals ($)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Residual Distribution", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.2)

    plt.suptitle(f"Residual Analysis — {model_name}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, f"residuals_{model_name.replace(' ', '_').lower()}")
    plt.close(fig)


def plot_feature_importance(
    model, feature_names: list, model_name: str, top_n: int = 30
) -> None:
    """Horizontal bar chart of top feature importances."""
    if not hasattr(model, "feature_importances_"):
        logger.info(f"{model_name} does not expose feature_importances_; skipping plot.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_imp      = importances[indices]

    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.35)))
    fig.patch.set_facecolor(DARK_BG)

    colors = plt.cm.plasma(np.linspace(0.3, 0.9, top_n))
    ax.barh(top_features[::-1], top_imp[::-1], color=colors[::-1], edgecolor="none")
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}",
                 fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    save_figure(fig, f"feature_importance_{model_name.replace(' ', '_').lower()}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameter Tuning
# ──────────────────────────────────────────────────────────────────────────────

def tune_model(
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_name: str,
) -> object:
    """
    Tune the best model using RandomizedSearchCV followed by GridSearchCV.

    Returns the best estimator found.
    """
    logger.info(f"Hyperparameter tuning: {model_name}")

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ── Parameter grids ──────────────────────────────────────────────────────
    param_grids = {
        "Random Forest": {
            "random": {
                "n_estimators":    [100, 200, 300, 500],
                "max_depth":       [None, 10, 15, 20],
                "min_samples_leaf":[1, 2, 4],
                "max_features":    ["sqrt", "log2", 0.5],
            },
            "grid": {
                "n_estimators": [200, 300],
                "max_depth":    [15, 20],
                "min_samples_leaf": [1, 2],
            },
        },
        "Gradient Boosting": {
            "random": {
                "n_estimators":  [200, 300, 400],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth":     [3, 4, 5, 6],
                "subsample":     [0.7, 0.8, 0.9],
            },
            "grid": {
                "n_estimators": [300, 400],
                "learning_rate":[0.05, 0.1],
                "max_depth":    [4, 5],
            },
        },
        "XGBoost": {
            "random": {
                "n_estimators":     [200, 300, 400],
                "learning_rate":    [0.01, 0.05, 0.1],
                "max_depth":        [4, 5, 6, 7],
                "subsample":        [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9],
                "reg_alpha":        [0, 0.1, 0.5],
                "reg_lambda":       [0.5, 1.0, 2.0],
            },
            "grid": {
                "n_estimators":  [300, 400],
                "learning_rate": [0.05, 0.1],
                "max_depth":     [5, 6],
            },
        },
        "LightGBM": {
            "random": {
                "n_estimators":     [200, 300, 400],
                "learning_rate":    [0.01, 0.05, 0.1],
                "max_depth":        [4, 5, 6, 7],
                "num_leaves":       [31, 50, 70],
                "subsample":        [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9],
            },
            "grid": {
                "n_estimators":  [300, 400],
                "learning_rate": [0.05, 0.1],
                "num_leaves":    [50, 70],
            },
        },
    }

    params = param_grids.get(model_name)
    if params is None:
        logger.warning(f"No tuning grid defined for {model_name}. Returning original model.")
        return model

    # RandomizedSearchCV — broad exploration
    rnd_search = RandomizedSearchCV(
        model,
        param_distributions=params["random"],
        n_iter=30,
        scoring="neg_root_mean_squared_error",
        cv=kf,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    with timer("RandomizedSearchCV"):
        rnd_search.fit(X_train, y_train)

    logger.info(f"RandomizedSearchCV best CV RMSE: {-rnd_search.best_score_:.6f}")
    logger.info(f"Best random params: {rnd_search.best_params_}")

    # GridSearchCV — fine-tune around best random params
    grid_search = GridSearchCV(
        rnd_search.best_estimator_,
        param_grid=params["grid"],
        scoring="neg_root_mean_squared_error",
        cv=kf,
        n_jobs=-1,
        verbose=1,
    )
    with timer("GridSearchCV"):
        grid_search.fit(X_train, y_train)

    logger.info(f"GridSearchCV best CV RMSE: {-grid_search.best_score_:.6f}")
    logger.info(f"Best grid params: {grid_search.best_params_}")

    return grid_search.best_estimator_


# ──────────────────────────────────────────────────────────────────────────────
# Main Training Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def train(data_path: str = None, seed: int = RANDOM_STATE) -> dict:
    """
    End-to-end training pipeline.

    Parameters
    ----------
    data_path : str — path to train.csv
    seed      : int — random seed

    Returns
    -------
    dict with keys: pipeline, results, best_model_name
    """
    set_seed(seed)

    if data_path is None:
        data_path = DATA_DIR / "train.csv"

    logger.info("=" * 60)
    logger.info("House Price Prediction — Training Pipeline")
    logger.info("=" * 60)

    # ── Load Data ────────────────────────────────────────────────────────────
    with timer("Data loading"):
        df = pd.read_csv(data_path)
        logger.info(f"Loaded: {df.shape}")

    # Extract target (log1p transform for normality)
    y = np.log1p(df["SalePrice"])
    X = df.drop(columns=["SalePrice"])

    # ── Train/Test Split ─────────────────────────────────────────────────────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    logger.info(f"Train: {X_train_raw.shape}, Test: {X_test_raw.shape}")

    # ── Preprocessing ────────────────────────────────────────────────────────
    with timer("Preprocessing"):
        preprocessor = HousePricePreprocessor()
        X_train_clean = preprocessor.fit_transform(X_train_raw)
        X_test_clean  = preprocessor.transform(X_test_raw)

    # ── Feature Engineering ──────────────────────────────────────────────────
    with timer("Feature engineering"):
        feat_eng = FeatureEngineer(scale=True)
        X_train = feat_eng.fit_transform(X_train_clean)
        X_test  = feat_eng.transform(X_test_clean)
    logger.info(f"Final features: {X_train.shape[1]}")

    # ── Train & Evaluate All Models ──────────────────────────────────────────
    models = get_models(seed)
    results = []
    trained_models = {}

    for name, model in models.items():
        logger.info(f"\n{'─' * 40}")
        logger.info(f"Training: {name}")
        with timer(name):
            model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        metrics["model"] = name
        results.append(metrics)
        trained_models[name] = model
        log_metrics(name, metrics)

    # ── Print Comparison Table ───────────────────────────────────────────────
    print("\n" + format_metrics_table(results))

    # ── Determine Best Model (by R²) ─────────────────────────────────────────
    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    best_name  = results_df.iloc[0]["model"]
    logger.info(f"\nBest model: {best_name} (R²={results_df.iloc[0]['R2']:.4f})")

    # ── Visualizations ───────────────────────────────────────────────────────
    plot_model_comparison(results)
    best_model = trained_models[best_name]
    plot_actual_vs_predicted(best_model, X_test, y_test.values, best_name)
    plot_residuals(best_model, X_test, y_test.values, best_name)
    plot_feature_importance(best_model, list(X_train.columns), best_name)

    # ── Hyperparameter Tuning ────────────────────────────────────────────────
    logger.info(f"\nTuning best model: {best_name}")
    tuned_model = tune_model(best_model, X_train, y_train, best_name)
    tuned_metrics = evaluate_model(tuned_model, X_train, y_train, X_test, y_test)
    logger.info(f"Tuned model metrics: {tuned_metrics}")
    plot_actual_vs_predicted(tuned_model, X_test, y_test.values, f"{best_name} (Tuned)")
    plot_residuals(tuned_model, X_test, y_test.values, f"{best_name} (Tuned)")
    plot_feature_importance(tuned_model, list(X_train.columns), f"{best_name} (Tuned)")

    # ── Save Pipeline & Model ────────────────────────────────────────────────
    pipeline = {
        "preprocessor":     preprocessor,
        "feature_engineer": feat_eng,
        "model":            tuned_model,
    }

    # Save full pipeline with joblib
    pipeline_path = MODELS_DIR / "house_price_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    logger.info(f"Pipeline saved → {pipeline_path}")

    # Save model separately with pickle
    model_path = MODELS_DIR / "house_price_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(tuned_model, f)
    logger.info(f"Model saved → {model_path}")

    # Also save with joblib for convenience
    model_jl_path = MODELS_DIR / "house_price_model.joblib"
    joblib.dump(tuned_model, model_jl_path)
    logger.info(f"Model saved → {model_jl_path}")

    logger.info("\nTraining pipeline complete!")
    return {
        "pipeline":        pipeline,
        "results":         results_df,
        "best_model_name": best_name,
        "trained_models":  trained_models,
        "feature_names":   list(X_train.columns),
        "X_train":         X_train,
        "X_test":          X_test,
        "y_train":         y_train,
        "y_test":          y_test,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train house price prediction models."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DATA_DIR / "train.csv"),
        help="Path to training CSV",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()
    train(data_path=args.data, seed=args.seed)


if __name__ == "__main__":
    main()
