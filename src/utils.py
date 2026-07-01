"""
utils.py — Shared Utility Functions
====================================
Provides reproducibility helpers, I/O utilities,
logging functions, and metric formatting used across
all modules in the House Price Prediction project.
"""

import os
import random
import logging
import time
from functools import wraps
from pathlib import Path
from contextlib import contextmanager

import numpy as np

# ─── Project Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
FIGURES_DIR  = OUTPUTS_DIR / "figures"
LOGS_DIR     = OUTPUTS_DIR / "logs"

# Ensure output directories exist
for _d in [MODELS_DIR, FIGURES_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Logging Setup ──────────────────────────────────────────────────────────────
def get_logger(name: str = "house_price") -> logging.Logger:
    """Return a configured logger that writes to console and a log file."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler
        log_file = LOGS_DIR / "training.log"
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

logger = get_logger()

# ─── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    logger.info(f"Global random seed set to {seed}")

# ─── Timing Utilities ──────────────────────────────────────────────────────────
@contextmanager
def timer(label: str = ""):
    """Context manager that logs elapsed time."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info(f"{label} completed in {elapsed:.2f}s")


def timeit(func):
    """Decorator that logs execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with timer(func.__name__):
            return func(*args, **kwargs)
    return wrapper

# ─── Figure Saving ─────────────────────────────────────────────────────────────
def save_figure(fig, name: str, dpi: int = 150) -> Path:
    """
    Save a matplotlib figure to the outputs/figures directory.

    Parameters
    ----------
    fig  : matplotlib.figure.Figure
    name : str — filename without extension
    dpi  : int — resolution

    Returns
    -------
    Path to saved file.
    """
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"Figure saved → {path}")
    return path

# ─── Metric Logging ────────────────────────────────────────────────────────────
def log_metrics(model_name: str, metrics: dict) -> None:
    """Pretty-print and log a dictionary of evaluation metrics."""
    border = "─" * 50
    logger.info(border)
    logger.info(f"  Model : {model_name}")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            logger.info(f"  {k:<25}: {v:.6f}")
        else:
            logger.info(f"  {k:<25}: {v}")
    logger.info(border)


def format_metrics_table(results: list[dict]) -> str:
    """
    Format a list of metric dicts as a printable table.

    Parameters
    ----------
    results : list of dicts with keys: model, R2, MAE, MSE, RMSE, CV_R2

    Returns
    -------
    Formatted string table.
    """
    header = f"{'Model':<30} {'R²':>8} {'MAE':>12} {'RMSE':>12} {'CV R²':>10}"
    divider = "─" * len(header)
    rows = [header, divider]
    for r in results:
        rows.append(
            f"{r['model']:<30} {r['R2']:>8.4f} {r['MAE']:>12.2f}"
            f" {r['RMSE']:>12.2f} {r.get('CV_R2', float('nan')):>10.4f}"
        )
    return "\n".join(rows)

# ─── Data Validation ───────────────────────────────────────────────────────────
def check_dataframe(df, name: str = "DataFrame") -> None:
    """Log basic stats about a DataFrame."""
    logger.info(f"{name}: shape={df.shape}, "
                f"nulls={df.isnull().sum().sum()}, "
                f"duplicates={df.duplicated().sum()}")
