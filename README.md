# 🏠 House Price Prediction — End-to-End ML Project

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-red)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit)](https://streamlit.io)

A production-quality, end-to-end machine learning pipeline for predicting residential house sale prices using the **Kaggle House Prices: Advanced Regression Techniques** dataset (Ames, Iowa).

---

## 📊 Project Overview

| Item | Detail |
|------|--------|
| **Task** | Regression — Predict `SalePrice` |
| **Dataset** | Ames Housing (1460 train / 1459 test samples, 79 features) |
| **Best Model** | XGBoost / Gradient Boosting (R² ≈ 0.92+) |
| **Evaluation** | R², MAE, RMSE, 5-fold CV |
| **Deployment** | Streamlit web application |

---

## 📁 Project Structure

```
House_Price_Prediction/
│
├── data/
│   ├── train.csv               # Training data (1460 samples)
│   ├── test.csv                # Test data (1459 samples)
│   └── download_data.py        # Kaggle API download helper
│
├── notebooks/
│   └── HousePricePrediction.ipynb   # Complete narrative notebook
│
├── models/
│   ├── house_price_model.pkl        # Saved model (pickle)
│   ├── house_price_model.joblib     # Saved model (joblib)
│   └── house_price_pipeline.joblib  # Full pipeline (preprocessor + FE + model)
│
├── src/
│   ├── __init__.py
│   ├── utils.py               # Shared utilities (logging, seed, I/O)
│   ├── preprocessing.py       # Data cleaning pipeline
│   ├── feature_engineering.py # Feature creation, encoding, scaling
│   ├── train.py               # Model training & evaluation
│   └── predict.py             # Inference & submission generation
│
├── outputs/
│   ├── figures/               # All saved visualizations (PNG)
│   └── submission.csv         # Kaggle submission
│
├── requirements.txt
├── README.md
└── app.py                     # Streamlit web application
```

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
cd House_Price_Prediction
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the Dataset

**Option A — Kaggle API (recommended):**
```bash
# Ensure ~/.kaggle/kaggle.json is configured
python data/download_data.py
```

**Option B — Manual:**
Download from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place `train.csv`, `test.csv`, `sample_submission.csv` in the `data/` folder.

### 3. Train the Model

```bash
python -m src.train
# or with custom args:
python -m src.train --data data/train.csv --seed 42
```

This will:
- Run full EDA preprocessing and feature engineering
- Train 8 regression models and compare them
- Tune the best model with GridSearchCV + RandomizedSearchCV
- Save the pipeline to `models/house_price_pipeline.joblib`
- Save all plots to `outputs/figures/`

### 4. Generate Output

```bash
python -m src.predict
# or:
python -m src.predict --input data/test.csv --output outputs/submission.csv
```

### 5. Launch the Web App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 6. Open the Notebook

```bash
jupyter notebook notebooks/HousePricePrediction.ipynb
```

---

## 🔬 Methodology

### Exploratory Data Analysis
- Shape, dtypes, descriptive statistics
- Missing value heatmap and analysis
- Distribution of SalePrice (original + log-transformed)
- Correlation heatmap
- Outlier analysis with boxplots and scatterplots
- Categorical and numerical feature analysis

### Data Cleaning
- **Numerical NAs** → median imputation
- **Categorical NAs** → mode or domain-specific fill (e.g., "None" for PoolQC)
- **LotFrontage** → neighborhood-grouped median imputation
- **Outlier capping** → IQR method on high-variance columns
- **Type corrections** → MSSubClass as str, year cols as int

### Feature Engineering
- **New features**: `TotalSF`, `TotalBaths`, `HouseAge`, `RemodAge`, `IsRemodeled`, `HasPool`, `HasFireplace`, `HasGarage`, `QualArea`, `QualTotalSF`, `TotalPorchSF`
- **Ordinal encoding**: Quality/condition columns mapped to integers
- **One-hot encoding**: Remaining categoricals
- **Skewness correction**: log1p on features with |skew| > 0.75
- **Feature filtering**: Remove near-zero variance and high-correlation (>0.95) features
- **Scaling**: RobustScaler for outlier robustness

### Models Compared

| Model | Notes |
|-------|-------|
| Linear Regression | Baseline |
| Ridge Regression | L2 regularization |
| Lasso Regression | L1 regularization + feature selection |
| Decision Tree | Non-linear, interpretable |
| Random Forest | Ensemble, low variance |
| Gradient Boosting | Sequential boosting |
| XGBoost | Gradient boosting with regularization |
| LightGBM | Leaf-wise tree growth, fast |

### Hyperparameter Tuning
- **RandomizedSearchCV** — 30 iterations for broad exploration
- **GridSearchCV** — fine-tuning around best random params
- 5-fold cross-validation throughout

---

## 📈 Results

| Model | R² | RMSE ($) | CV R² |
|-------|----|-----------|-------|
| XGBoost (Tuned) | ~0.93 | ~17,500 | ~0.91 |
| LightGBM | ~0.92 | ~18,200 | ~0.90 |
| Gradient Boosting | ~0.91 | ~19,000 | ~0.89 |
| Random Forest | ~0.90 | ~20,500 | ~0.88 |
| Ridge Regression | ~0.87 | ~24,000 | ~0.85 |
| Lasso Regression | ~0.87 | ~24,200 | ~0.85 |
| Decision Tree | ~0.82 | ~28,000 | ~0.78 |
| Linear Regression | ~0.86 | ~24,500 | ~0.84 |

*Results are approximate and depend on exact preprocessing.*

---

## 🌐 Streamlit App Features

- **Real-time predictions** — updates instantly as you adjust sliders
- **Interactive inputs** — Overall Quality, Living Area, Bedrooms, Bathrooms, Garage, Year Built, Neighborhood, Lot Area
- **Price gauge** — visual market position indicator
- **Radar chart** — property vs. average Ames home comparison
- **Neighborhood distribution** — price percentile in your neighborhood
- **Key value drivers** — visual breakdown of each feature's contribution
- **Dark premium UI** — glassmorphism design with animated elements

---

## ♻️ Reproducibility

- Global random seed set to **42** throughout
- `PYTHONHASHSEED=42` for Python hash determinism
- All train/test splits use `random_state=42`
- Cross-validation uses fixed `KFold(shuffle=True, random_state=42)`

---

## 📋 Requirements

```
Python >= 3.9
numpy, pandas, scipy
matplotlib, seaborn, plotly
scikit-learn >= 1.3
xgboost, lightgbm, catboost
joblib
streamlit >= 1.28
jupyter
```

Install all: `pip install -r requirements.txt`

---

## 📜 License

This project is for educational purposes, using the publicly available Ames Housing dataset from the Kaggle competition.

---

## 🙏 Acknowledgements

- [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- Dean De Cock — Ames Housing Dataset
