"""
app.py — Streamlit House Price Prediction Web Application
==========================================================
Interactive web app for predicting residential house sale prices.
Users can input property features via sliders and dropdowns
and see the predicted price instantly.

Run:
    streamlit run app.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Add project root to path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(124, 92, 191, 0.3);
    }

    [data-testid="stSidebar"] .stSlider > div > div > div > div {
        background: #7c5cbf !important;
    }

    /* ── Header ── */
    .main-header {
        background: linear-gradient(135deg, rgba(124, 92, 191, 0.15) 0%, rgba(233, 108, 125, 0.1) 100%);
        border: 1px solid rgba(124, 92, 191, 0.3);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }

    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7c5cbf, #e96c7d, #5bc0eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        padding: 0;
    }

    .main-header p {
        color: rgba(255,255,255,0.6);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    /* ── Price Card ── */
    .price-card {
        background: linear-gradient(135deg, rgba(124, 92, 191, 0.2), rgba(233, 108, 125, 0.15));
        border: 2px solid rgba(124, 92, 191, 0.5);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        backdrop-filter: blur(20px);
        box-shadow: 0 20px 60px rgba(124, 92, 191, 0.2);
        transition: all 0.3s ease;
    }

    .price-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 30px 80px rgba(124, 92, 191, 0.35);
    }

    .price-label {
        color: rgba(255,255,255,0.6);
        font-size: 1rem;
        font-weight: 500;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .price-value {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7c5cbf, #e96c7d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
    }

    .price-range {
        color: rgba(255,255,255,0.4);
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(124, 92, 191, 0.2);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.25s ease;
    }

    .metric-card:hover {
        background: rgba(124, 92, 191, 0.1);
        border-color: rgba(124, 92, 191, 0.5);
        transform: translateY(-3px);
    }

    .metric-label {
        color: rgba(255,255,255,0.5);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 500;
    }

    .metric-value {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }

    /* ── Section Headers ── */
    .section-header {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        font-weight: 700;
        border-left: 4px solid #7c5cbf;
        padding-left: 1rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* ── Sidebar Inputs ── */
    .sidebar-section {
        background: rgba(124, 92, 191, 0.08);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(124, 92, 191, 0.15);
    }

    /* ── Feature Bar ── */
    .feature-bar-container {
        margin: 0.4rem 0;
    }

    /* ── Warning banner ── */
    .warning-banner {
        background: linear-gradient(135deg, rgba(255, 165, 0, 0.15), rgba(255, 100, 0, 0.1));
        border: 1px solid rgba(255, 165, 0, 0.4);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.95rem;
    }

    /* ── Streamlit element overrides ── */
    .stSlider > label { color: rgba(255,255,255,0.8) !important; }
    .stSelectbox > label { color: rgba(255,255,255,0.8) !important; }
    .stNumberInput > label { color: rgba(255,255,255,0.8) !important; }
    div[data-testid="stMetricValue"] { color: #7c5cbf !important; }
    .stButton > button {
        background: linear-gradient(135deg, #7c5cbf, #e96c7d);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(124, 92, 191, 0.4);
    }
    hr { border-color: rgba(124, 92, 191, 0.2) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Constants ────────────────────────────────────────────────────────────────
PIPELINE_PATH = PROJECT_ROOT / "models" / "house_price_pipeline.joblib"

NEIGHBORHOODS = [
    "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr",
    "Crawfor", "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel",
    "NAmes", "NoRidge", "NPkVill", "NridgHt", "NWAmes", "OldTown",
    "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker",
]

NEIGHBORHOOD_TIER = {
    "NoRidge": "Premium", "NridgHt": "Premium", "StoneBr": "Premium",
    "Veenker": "Premium", "Timber": "Premium", "Somerst": "Premium",
    "ClearCr": "Upper Mid", "Crawfor": "Upper Mid", "CollgCr": "Upper Mid",
    "Gilbert": "Upper Mid", "NWAmes": "Upper Mid",
    "Mitchel": "Mid", "NAmes": "Mid", "SawyerW": "Mid", "Sawyer": "Mid",
    "Edwards": "Mid", "BrkSide": "Lower Mid", "OldTown": "Lower Mid",
    "IDOTRR": "Lower Mid", "MeadowV": "Budget", "BrDale": "Budget",
}

NEIGHBORHOOD_MEDIANS = {
    "NoRidge": 335000, "NridgHt": 315000, "StoneBr": 278000,
    "Veenker": 250000, "Timber": 240000, "Somerst": 225000,
    "ClearCr": 212000, "Crawfor": 210000, "CollgCr": 197000,
    "Gilbert": 181000, "NWAmes": 182000, "Mitchel": 159000,
    "NAmes": 140000, "SawyerW": 179000, "Sawyer": 135000,
    "Edwards": 130000, "BrkSide": 124000, "OldTown": 121000,
    "IDOTRR": 104000, "MeadowV": 99000, "BrDale": 103000,
    "Blmngtn": 190000, "Blueste": 137000, "NPkVill": 145000,
    "SWISU": 133000,
}

# ─── Pipeline Loader ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading prediction model…")
def load_pipeline():
    if not PIPELINE_PATH.exists():
        return None
    try:
        return joblib.load(PIPELINE_PATH)
    except Exception as e:
        return None


# ─── Feature Input Builder ────────────────────────────────────────────────────

def build_input_row(inputs: dict) -> pd.DataFrame:
    """
    Construct a single-row DataFrame from user inputs,
    with enough columns to pass through the preprocessing pipeline.
    This creates all the features expected by the trained model
    while filling unknown columns with reasonable defaults.
    """
    defaults = {
        # Lot / Location
        "LotFrontage": float(inputs["lot_area"] ** 0.5),
        "LotArea": inputs["lot_area"],
        "Street": "Pave",
        "Alley": "None",
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "Utilities": "AllPub",
        "LotConfig": "Inside",
        "LandSlope": "Gtl",
        "Neighborhood": inputs["neighborhood"],
        "Condition1": "Norm",
        "Condition2": "Norm",

        # Building
        "BldgType": "1Fam",
        "HouseStyle": "2Story" if inputs["year_built"] > 1990 else "1Story",
        "OverallQual": inputs["overall_qual"],
        "OverallCond": 5,
        "YearBuilt": inputs["year_built"],
        "YearRemodAdd": inputs["year_built"],
        "RoofStyle": "Gable",
        "RoofMatl": "CompShg",
        "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd",
        "MasVnrType": "None",
        "MasVnrArea": 0,
        "ExterQual": "TA",
        "ExterCond": "TA",
        "Foundation": "PConc",

        # Basement
        "BsmtQual": "TA",
        "BsmtCond": "TA",
        "BsmtExposure": "No",
        "BsmtFinType1": "Unf",
        "BsmtFinSF1": 0,
        "BsmtFinType2": "Unf",
        "BsmtFinSF2": 0,
        "BsmtUnfSF": 0,
        "TotalBsmtSF": int(inputs["gr_liv_area"] * 0.6),

        # HVAC
        "Heating": "GasA",
        "HeatingQC": "Ex",
        "CentralAir": "Y",
        "Electrical": "SBrkr",

        # Rooms / Living
        "1stFlrSF": int(inputs["gr_liv_area"] * 0.6),
        "2ndFlrSF": int(inputs["gr_liv_area"] * 0.4),
        "LowQualFinSF": 0,
        "GrLivArea": inputs["gr_liv_area"],
        "BsmtFullBath": 0,
        "BsmtHalfBath": 0,
        "FullBath": inputs["full_bath"],
        "HalfBath": inputs["half_bath"],
        "BedroomAbvGr": inputs["bedrooms"],
        "KitchenAbvGr": 1,
        "KitchenQual": "Gd",
        "TotRmsAbvGrd": inputs["bedrooms"] + inputs["full_bath"] + 3,
        "Functional": "Typ",
        "Fireplaces": inputs["fireplaces"],
        "FireplaceQu": "Gd" if inputs["fireplaces"] > 0 else "None",

        # Garage
        "GarageType": "Attchd" if inputs["garage_cars"] > 0 else "None",
        "GarageYrBlt": inputs["year_built"],
        "GarageFinish": "Fin" if inputs["garage_cars"] > 0 else "None",
        "GarageCars": inputs["garage_cars"],
        "GarageArea": inputs["garage_cars"] * 300,
        "GarageQual": "TA" if inputs["garage_cars"] > 0 else "None",
        "GarageCond": "TA" if inputs["garage_cars"] > 0 else "None",
        "PavedDrive": "Y",

        # Outdoor
        "WoodDeckSF": 0,
        "OpenPorchSF": 0,
        "EnclosedPorch": 0,
        "3SsnPorch": 0,
        "ScreenPorch": 0,
        "PoolArea": 0,
        "PoolQC": "None",
        "Fence": "None",
        "MiscFeature": "None",
        "MiscVal": 0,

        # Sale
        "MoSold": 6,
        "YrSold": 2010,
        "SaleType": "WD",
        "SaleCondition": "Normal",

        # MSSubClass / MSZoning
        "MSSubClass": 60,
        "MSZoning": "RL",
    }
    return pd.DataFrame([defaults])


# ─── Gauge Chart ──────────────────────────────────────────────────────────────

def make_price_gauge(price: float) -> go.Figure:
    """Plotly gauge chart showing price relative to market range."""
    min_p, max_p = 50_000, 800_000
    pct = min(max((price - min_p) / (max_p - min_p), 0), 1)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=price,
        number={"prefix": "$", "valueformat": ",.0f",
                "font": {"size": 28, "color": "#ffffff", "family": "Inter"}},
        delta={"reference": 180_000, "valueformat": ",.0f",
               "increasing": {"color": "#4caf50"}, "decreasing": {"color": "#e96c7d"}},
        gauge={
            "axis": {
                "range": [min_p, max_p],
                "tickformat": "$,.0f",
                "tickfont": {"color": "#aaaaaa", "size": 9},
                "nticks": 6,
            },
            "bar": {"color": "rgba(124, 92, 191, 0.9)", "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [min_p, 150_000],    "color": "rgba(91,192,235,0.15)"},
                {"range": [150_000, 300_000],   "color": "rgba(76,175,80,0.15)"},
                {"range": [300_000, 500_000],   "color": "rgba(255,165,0,0.15)"},
                {"range": [500_000, max_p],     "color": "rgba(233,108,125,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#e96c7d", "width": 3},
                "thickness": 0.8,
                "value": 180_000,
            },
        },
        title={"text": "Market Position", "font": {"color": "rgba(255,255,255,0.6)", "size": 13}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=20, r=20, t=30, b=10),
        font={"family": "Inter"},
    )
    return fig


# ─── Feature Comparison Chart ─────────────────────────────────────────────────

def make_radar_chart(inputs: dict) -> go.Figure:
    """Radar chart comparing user inputs to dataset averages."""
    categories = ["Quality", "Living Area", "Bedrooms", "Bathrooms", "Garage", "House Age"]

    # Normalize to 0-10 scale
    current_year = 2024
    user_vals = [
        inputs["overall_qual"],
        min(inputs["gr_liv_area"] / 5000 * 10, 10),
        min(inputs["bedrooms"] / 6 * 10, 10),
        min((inputs["full_bath"] + 0.5 * inputs["half_bath"]) / 4 * 10, 10),
        min(inputs["garage_cars"] / 3 * 10, 10),
        max(10 - (current_year - inputs["year_built"]) / 12, 0),
    ]

    avg_vals = [6.1, 4.1, 5.0, 4.5, 5.5, 4.5]  # Ames dataset approx averages

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=avg_vals + [avg_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Avg Ames Home",
        line=dict(color="rgba(255,255,255,0.3)", width=1),
        fillcolor="rgba(255,255,255,0.05)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=user_vals + [user_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Your Property",
        line=dict(color="#7c5cbf", width=2.5),
        fillcolor="rgba(124, 92, 191, 0.25)",
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 10],
                gridcolor="rgba(255,255,255,0.1)",
                tickfont=dict(color="rgba(255,255,255,0.5)", size=9),
                tickvals=[2, 4, 6, 8, 10],
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.15)",
                tickfont=dict(color="rgba(255,255,255,0.8)", size=11),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            font=dict(color="rgba(255,255,255,0.7)", size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=340,
        margin=dict(l=60, r=60, t=30, b=30),
        font={"family": "Inter"},
    )
    return fig


# ─── Price History Chart ──────────────────────────────────────────────────────

def make_price_range_chart(predicted_price: float, neighborhood: str) -> go.Figure:
    """Show price percentile among neighborhood comps."""
    nbhd_median = NEIGHBORHOOD_MEDIANS.get(neighborhood, 175_000)
    prices = np.random.default_rng(hash(neighborhood) % 2**32).normal(
        loc=nbhd_median, scale=nbhd_median * 0.22, size=200
    )
    prices = np.clip(prices, 50_000, 900_000)

    percentile = (prices < predicted_price).mean() * 100

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=prices,
        nbinsx=25,
        name=f"{neighborhood} homes",
        marker=dict(color="rgba(91,192,235,0.3)", line=dict(color="rgba(91,192,235,0.6)", width=0.5)),
        hovertemplate="Price: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(
        x=nbhd_median, line_dash="dash",
        line_color="rgba(255,255,255,0.35)", line_width=2,
        annotation_text="Median", annotation_font_color="rgba(255,255,255,0.5)",
    )
    fig.add_vline(
        x=predicted_price, line_dash="solid",
        line_color="#7c5cbf", line_width=3,
        annotation_text=f"Your Property<br>{percentile:.0f}th pct.",
        annotation_font_color="#b39ddb",
        annotation_font_size=12,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,26,46,0.7)",
        xaxis=dict(
            title="Sale Price ($)", tickformat="$,.0f",
            color="rgba(255,255,255,0.6)", gridcolor="rgba(255,255,255,0.07)",
        ),
        yaxis=dict(
            title="# of Properties",
            color="rgba(255,255,255,0.6)", gridcolor="rgba(255,255,255,0.07)",
        ),
        font=dict(family="Inter", color="rgba(255,255,255,0.7)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=280,
        margin=dict(l=50, r=20, t=15, b=50),
        showlegend=False,
    )
    return fig, percentile


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🏠 House Price Predictor</h1>
        <p>AI-powered valuation using the Ames Housing dataset · Gradient Boosting / XGBoost model</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load Pipeline ──────────────────────────────────────────────────────────
    pipeline = load_pipeline()
    pipeline_loaded = pipeline is not None

    if not pipeline_loaded:
        st.markdown("""
        <div class="warning-banner">
            ⚠️ <strong>Model not found.</strong> Please train the model first by running:<br>
            <code style="background:rgba(255,255,255,0.1);padding:3px 8px;border-radius:5px;">
                python -m src.train
            </code><br>
            In the meantime, you can explore the interface — predictions will use a
            statistical estimate based on neighborhood medians and feature correlations.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

    # ── Sidebar Inputs ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Property Details")
        st.markdown("Adjust the sliders to configure your property.")
        st.markdown("---")

        st.markdown("### 📍 Location")
        neighborhood = st.selectbox(
            "Neighborhood",
            options=sorted(NEIGHBORHOODS),
            index=sorted(NEIGHBORHOODS).index("CollgCr"),
            help="Select the neighborhood in Ames, Iowa",
        )
        tier = NEIGHBORHOOD_TIER.get(neighborhood, "Mid")
        tier_colors = {
            "Premium": "🟣", "Upper Mid": "🔵", "Mid": "🟢",
            "Lower Mid": "🟡", "Budget": "🔴",
        }
        st.caption(f"{tier_colors.get(tier, '⚪')} Tier: **{tier}** neighborhood")

        st.markdown("### 🏗️ Structure")
        overall_qual = st.slider(
            "Overall Quality (1–10)", min_value=1, max_value=10, value=7,
            help="1=Very Poor, 10=Excellent",
        )
        year_built = st.slider(
            "Year Built", min_value=1872, max_value=2010, value=2000, step=1,
        )
        gr_liv_area = st.slider(
            "Above Ground Living Area (sq ft)",
            min_value=300, max_value=6000, value=1800, step=50,
        )
        lot_area = st.slider(
            "Lot Area (sq ft)",
            min_value=1300, max_value=215000, value=9000, step=100,
        )

        st.markdown("### 🛏️ Rooms")
        col1, col2 = st.columns(2)
        with col1:
            bedrooms = st.number_input(
                "Bedrooms", min_value=0, max_value=10, value=3, step=1,
            )
            full_bath = st.number_input(
                "Full Baths", min_value=0, max_value=6, value=2, step=1,
            )
        with col2:
            half_bath = st.number_input(
                "Half Baths", min_value=0, max_value=4, value=0, step=1,
            )
            fireplaces = st.number_input(
                "Fireplaces", min_value=0, max_value=4, value=1, step=1,
            )

        st.markdown("### 🚗 Garage")
        garage_cars = st.slider(
            "Garage Capacity (cars)", min_value=0, max_value=4, value=2,
        )

        st.markdown("---")
        predict_btn = st.button("🔮 Predict Price", use_container_width=True)

    # ── Collect Inputs ────────────────────────────────────────────────────────
    inputs = {
        "overall_qual": overall_qual,
        "gr_liv_area":  gr_liv_area,
        "bedrooms":     bedrooms,
        "full_bath":    full_bath,
        "half_bath":    half_bath,
        "garage_cars":  garage_cars,
        "year_built":   year_built,
        "neighborhood": neighborhood,
        "lot_area":     lot_area,
        "fireplaces":   fireplaces,
    }

    # ── Predict ───────────────────────────────────────────────────────────────
    def get_prediction(inputs: dict) -> float:
        if pipeline_loaded:
            try:
                row = build_input_row(inputs)
                X = pipeline["preprocessor"].transform(row)
                X = pipeline["feature_engineer"].transform(X)
                y_log = pipeline["model"].predict(X)
                return float(np.expm1(y_log[0]))
            except Exception as e:
                st.warning(f"Pipeline error: {e}. Using fallback estimate.")

        # Statistical fallback (no model)
        base = NEIGHBORHOOD_MEDIANS.get(inputs["neighborhood"], 175_000)
        price = base
        price *= 1 + (inputs["overall_qual"] - 6) * 0.12
        price *= 1 + (inputs["gr_liv_area"] - 1500) / 1500 * 0.35
        price *= 1 + (inputs["bedrooms"] - 3) * 0.03
        price *= 1 + (inputs["full_bath"] - 2) * 0.04
        price *= 1 + inputs["garage_cars"] * 0.03
        price *= 1 + (inputs["year_built"] - 1990) / 1990 * 0.4
        return max(price, 30_000)

    predicted_price = get_prediction(inputs)
    low_est = predicted_price * 0.92
    high_est = predicted_price * 1.08

    # ── Layout: 3 columns ────────────────────────────────────────────────────
    col_price, col_radar = st.columns([1, 1], gap="large")

    with col_price:
        # ── Price Card ───────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="price-card">
            <div class="price-label">Estimated Sale Price</div>
            <div class="price-value">${predicted_price:,.0f}</div>
            <div class="price-range">
                Range: ${low_est:,.0f} — ${high_est:,.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

        # ── Gauge Chart ──────────────────────────────────────────────────────
        st.plotly_chart(make_price_gauge(predicted_price), use_container_width=True)

        # ── Key Metrics ──────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📊 Property Summary</div>', unsafe_allow_html=True)
        price_per_sqft = predicted_price / max(gr_liv_area, 1)
        house_age = 2024 - year_built
        total_baths = full_bath + 0.5 * half_bath
        total_sf = gr_liv_area + int(gr_liv_area * 0.6)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">$/sq ft</div>
                <div class="metric-value">${price_per_sqft:.0f}</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">House Age</div>
                <div class="metric-value">{house_age} yrs</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Baths</div>
                <div class="metric-value">{total_baths:.1f}</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Est. Total SF</div>
                <div class="metric-value">{total_sf:,}</div>
            </div>""", unsafe_allow_html=True)

    with col_radar:
        st.markdown('<div class="section-header">🎯 Property vs. Average</div>', unsafe_allow_html=True)
        st.plotly_chart(make_radar_chart(inputs), use_container_width=True)

        st.markdown('<div class="section-header">🏘️ Neighborhood Price Distribution</div>', unsafe_allow_html=True)
        price_chart, percentile = make_price_range_chart(predicted_price, neighborhood)
        st.plotly_chart(price_chart, use_container_width=True)
        pct_color = "#4caf50" if percentile >= 50 else "#e96c7d"
        st.markdown(f"""
        <p style="text-align:center;color:rgba(255,255,255,0.6);font-size:0.9rem;">
            Your property is in the
            <strong style="color:{pct_color}">{percentile:.0f}th percentile</strong>
            for <strong>{neighborhood}</strong> neighborhood.
        </p>
        """, unsafe_allow_html=True)

    # ── Feature Influence Table ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🔍 Key Value Drivers</div>', unsafe_allow_html=True)

    drivers = [
        ("Overall Quality",    f"{overall_qual}/10",  overall_qual / 10,    "#7c5cbf"),
        ("Living Area",        f"{gr_liv_area:,} sq ft", min(gr_liv_area/6000, 1), "#5bc0eb"),
        ("Neighborhood Tier",  tier,                  {"Premium": 1.0, "Upper Mid": 0.8, "Mid": 0.6, "Lower Mid": 0.4, "Budget": 0.2}.get(tier, 0.5), "#e96c7d"),
        ("Year Built",         str(year_built),       min((year_built - 1872) / (2010 - 1872), 1), "#ffa500"),
        ("Garage Capacity",    f"{garage_cars} car(s)", garage_cars / 4,    "#4caf50"),
        ("Bathrooms",          f"{full_bath + 0.5*half_bath:.1f}",  min((full_bath + 0.5*half_bath) / 4, 1), "#ab47bc"),
    ]

    driver_cols = st.columns(2)
    for i, (label, value, pct, color) in enumerate(drivers):
        with driver_cols[i % 2]:
            bar_w = max(int(pct * 100), 3)
            st.markdown(f"""
            <div style="margin:0.5rem 0;">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="color:rgba(255,255,255,0.75);font-size:0.88rem;">{label}</span>
                    <span style="color:rgba(255,255,255,0.9);font-size:0.88rem;font-weight:600;">{value}</span>
                </div>
                <div style="background:rgba(255,255,255,0.07);border-radius:4px;height:6px;">
                    <div style="width:{bar_w}%;background:{color};border-radius:4px;height:6px;
                                transition:width 0.5s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <p style="text-align:center;color:rgba(255,255,255,0.3);font-size:0.8rem;">
        🏠 House Price Predictor · Trained on Ames Housing Dataset ·
        Model: Gradient Boosting / XGBoost · For educational purposes only
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
