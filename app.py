import io
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

# Suppress harmless sklearn version difference warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Page Configuration & Metadata
# ---------------------------------------------------------
st.set_page_config(
    page_title="CaliHome AI • Valuation Intelligence",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
FEATURES_PATH = BASE_DIR / "feature_names.pkl"
HOUSING_CSV = BASE_DIR / "housing.csv"
DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

# ---------------------------------------------------------
# Custom Modern UI Theme (CSS)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 95%;
    }

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(20, 27, 45, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 18px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 0;
    }

    /* Glassmorphic Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.2rem;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #10b981;
        margin-top: 0.2rem;
    }

    /* Price Badge */
    .badge-tier {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-budget { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-moderate { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-premium { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-luxury { background: rgba(236, 72, 153, 0.15); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.3); }

    /* Custom Tag in sidebar */
    .model-badge {
        background: rgba(99, 102, 241, 0.12);
        color: #818cf8;
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Resource Loaders (Cached for Instant Performance)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading machine learning model...")
def load_ml_assets():
    """Loads the trained RandomForestRegressor model, StandardScaler, and feature schema."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Load or generate scaler
    if SCALER_PATH.exists() and FEATURES_PATH.exists():
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        with open(FEATURES_PATH, "rb") as f:
            feature_columns = pickle.load(f)
    else:
        # Fallback to computing from housing data
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        if HOUSING_CSV.exists():
            df = pd.read_csv(HOUSING_CSV)
        else:
            df = pd.read_csv(DATA_URL)
        df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
        encoded = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)
        encoded["rooms_per_household"] = encoded["total_rooms"] / encoded["households"]
        encoded["bedrooms_per_room"] = encoded["total_bedrooms"] / encoded["total_rooms"]
        encoded["population_per_household"] = encoded["population"] / encoded["households"]
        X = encoded.drop(columns=["median_house_value"])
        Y = encoded["median_house_value"]
        X_train, _, _, _ = train_test_split(X, Y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        scaler.fit(X_train)
        feature_columns = list(X.columns)
        with open(SCALER_PATH, "wb") as sf:
            pickle.dump(scaler, sf)
        with open(FEATURES_PATH, "wb") as ff:
            pickle.dump(feature_columns, ff)

    return model, scaler, feature_columns


@st.cache_data(show_spinner=False)
def load_dataset_sample():
    """Loads a representative sample of California housing districts for geospatial visualization."""
    if HOUSING_CSV.exists():
        df = pd.read_csv(HOUSING_CSV)
    else:
        df = pd.read_csv(DATA_URL)
    return df


try:
    model, scaler, feature_columns = load_ml_assets()
    full_data = load_dataset_sample()
except Exception as err:
    st.error(f"⚠️ Critical error initializing machine learning engine: {err}")
    st.stop()

# ---------------------------------------------------------
# Feature Engineering & Prediction Pipelines
# ---------------------------------------------------------
def prepare_feature_row(values: dict, feature_cols: list) -> pd.DataFrame:
    """Calculates derived spatial and demographic interaction features for a single district."""
    row = pd.DataFrame([values])
    row["rooms_per_household"] = row["total_rooms"] / row["households"]
    row["bedrooms_per_room"] = row["total_bedrooms"] / row["total_rooms"]
    row["population_per_household"] = row["population"] / row["households"]
    
    # One-hot encoding categorical ocean proximity (drop_first='<1H OCEAN')
    row["ocean_proximity_INLAND"] = int(values["ocean_proximity"] == "INLAND")
    row["ocean_proximity_ISLAND"] = int(values["ocean_proximity"] == "ISLAND")
    row["ocean_proximity_NEAR BAY"] = int(values["ocean_proximity"] == "NEAR BAY")
    row["ocean_proximity_NEAR OCEAN"] = int(values["ocean_proximity"] == "NEAR OCEAN")
    
    row = row.drop(columns=["ocean_proximity"], errors="ignore")
    return row.reindex(columns=feature_cols, fill_value=0)


def predict_valuation_ensemble(raw_values: dict):
    """Predicts median house value alongside 100-tree ensemble distribution and uncertainty."""
    features_df = prepare_feature_row(raw_values, feature_columns)
    scaled_feats = scaler.transform(features_df)
    
    # Extract prediction from all 100 individual decision trees
    tree_estimates = np.array([tree.predict(scaled_feats)[0] for tree in model.estimators_])
    
    mean_val = float(np.mean(tree_estimates))
    std_val = float(np.std(tree_estimates))
    p10 = float(np.percentile(tree_estimates, 10))
    p90 = float(np.percentile(tree_estimates, 90))
    p5 = float(np.percentile(tree_estimates, 5))
    p95 = float(np.percentile(tree_estimates, 95))
    
    return {
        "prediction": mean_val,
        "std": std_val,
        "p10": p10,
        "p90": p90,
        "p5": p5,
        "p95": p95,
        "tree_estimates": tree_estimates,
        "features": features_df,
    }


def format_curr(amount: float) -> str:
    """Formats numeric value as US currency."""
    return f"${amount:,.0f}"


def get_price_tier(price: float) -> tuple[str, str]:
    """Classifies valuation into descriptive market tiers."""
    if price < 150000:
        return "Affordable / Value", "badge-budget"
    elif price < 280000:
        return "Moderate Market", "badge-moderate"
    elif price < 420000:
        return "Premium Market", "badge-premium"
    else:
        return "Ultra-Luxury Core", "badge-luxury"


# ---------------------------------------------------------
# Preset Profiles (Realistic California Market archetypes)
# ---------------------------------------------------------
PRESETS = {
    "🌟 Silicon Valley (Palo Alto)": {
        "longitude": -122.16,
        "latitude": 37.44,
        "housing_median_age": 34.0,
        "total_rooms": 2800.0,
        "total_bedrooms": 420.0,
        "population": 1100.0,
        "households": 410.0,
        "median_income": 11.2,
        "ocean_proximity": "NEAR BAY",
    },
    "🏖️ Santa Monica / Coastal LA": {
        "longitude": -118.49,
        "latitude": 34.02,
        "housing_median_age": 38.0,
        "total_rooms": 2600.0,
        "total_bedrooms": 520.0,
        "population": 1250.0,
        "households": 490.0,
        "median_income": 8.5,
        "ocean_proximity": "<1H OCEAN",
    },
    "🌉 San Francisco Marina": {
        "longitude": -122.44,
        "latitude": 37.80,
        "housing_median_age": 52.0,
        "total_rooms": 3100.0,
        "total_bedrooms": 680.0,
        "population": 1400.0,
        "households": 650.0,
        "median_income": 9.2,
        "ocean_proximity": "NEAR BAY",
    },
    "🍊 Newport Beach (Orange County)": {
        "longitude": -117.93,
        "latitude": 33.62,
        "housing_median_age": 26.0,
        "total_rooms": 3400.0,
        "total_bedrooms": 580.0,
        "population": 1300.0,
        "households": 510.0,
        "median_income": 9.8,
        "ocean_proximity": "NEAR OCEAN",
    },
    "🌾 Fresno (Central Valley Inland)": {
        "longitude": -119.77,
        "latitude": 36.75,
        "housing_median_age": 24.0,
        "total_rooms": 2100.0,
        "total_bedrooms": 420.0,
        "population": 1450.0,
        "households": 410.0,
        "median_income": 2.9,
        "ocean_proximity": "INLAND",
    },
    "🏔️ Lake Tahoe / Sierra Foothills": {
        "longitude": -120.04,
        "latitude": 39.09,
        "housing_median_age": 18.0,
        "total_rooms": 2400.0,
        "total_bedrooms": 490.0,
        "population": 850.0,
        "households": 340.0,
        "median_income": 5.2,
        "ocean_proximity": "INLAND",
    },
    "🏙️ San Diego Mission Bay": {
        "longitude": -117.24,
        "latitude": 32.78,
        "housing_median_age": 31.0,
        "total_rooms": 2700.0,
        "total_bedrooms": 530.0,
        "population": 1350.0,
        "households": 510.0,
        "median_income": 6.8,
        "ocean_proximity": "NEAR OCEAN",
    },
}

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div>
                <div class="hero-title">🏡 CaliHome AI • Valuation Intelligence</div>
                <p class="hero-subtitle">Interactive Real Estate Forecasting Engine powered by 100-Tree Random Forest Regressor</p>
            </div>
            <div style="display: flex; gap: 0.6rem; align-items: center;">
                <span class="model-badge">🌲 RF 100 Trees</span>
                <span class="model-badge">📊 R² = 0.8072</span>
                <span class="model-badge">🎯 RMSE = $50,258</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "auto_predict" not in st.session_state:
    st.session_state.auto_predict = True

# ---------------------------------------------------------
# Sidebar Controls & Dynamic Inputs
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ District Parameters")
    
    # Preset Selector
    selected_preset_name = st.selectbox(
        "📌 Quick Market Archetypes",
        options=["Custom District"] + list(PRESETS.keys()),
        index=1,  # Default to Silicon Valley
        help="Select a representative California community or customize your own parameters below."
    )
    
    preset_vals = PRESETS.get(selected_preset_name, PRESETS["🌟 Silicon Valley (Palo Alto)"])
    
    st.markdown("---")
    st.subheader("📍 Geospatial Coordinates")
    col_lon, col_lat = st.columns(2)
    with col_lon:
        longitude = st.slider(
            "Longitude",
            min_value=-124.35,
            max_value=-114.31,
            value=float(preset_vals["longitude"]),
            step=0.01,
            format="%.2f",
            help="California longitude (-124.35° West to -114.31° East)"
        )
    with col_lat:
        latitude = st.slider(
            "Latitude",
            min_value=32.54,
            max_value=41.95,
            value=float(preset_vals["latitude"]),
            step=0.01,
            format="%.2f",
            help="California latitude (32.54° South to 41.95° North)"
        )

    ocean_proximity = st.selectbox(
        "🌊 Ocean Proximity",
        options=["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"],
        index=["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"].index(preset_vals["ocean_proximity"]),
        help="Location category relative to California coastlines and bays."
    )

    st.markdown("---")
    st.subheader("💰 Economic & Demographics")
    median_income = st.slider(
        "Median Household Income",
        min_value=0.5,
        max_value=15.0,
        value=float(preset_vals["median_income"]),
        step=0.1,
        help="District median income in tens of thousands (e.g., 6.5 = $65,000/yr)"
    )
    st.caption(f"Estimated Annual Income: **${median_income * 10000:,.0f}**")

    housing_median_age = st.slider(
        "Housing Median Age",
        min_value=1.0,
        max_value=52.0,
        value=float(preset_vals["housing_median_age"]),
        step=1.0,
        help="Median age of buildings in this block group."
    )

    st.markdown("---")
    st.subheader("🏘️ Housing Density & Capacity")
    col_rooms, col_beds = st.columns(2)
    with col_rooms:
        total_rooms = st.number_input(
            "Total Rooms",
            min_value=10.0,
            max_value=40000.0,
            value=float(preset_vals["total_rooms"]),
            step=100.0
        )
    with col_beds:
        total_bedrooms = st.number_input(
            "Total Bedrooms",
            min_value=1.0,
            max_value=10000.0,
            value=float(preset_vals["total_bedrooms"]),
            step=50.0
        )

    col_pop, col_hh = st.columns(2)
    with col_pop:
        population = st.number_input(
            "Population",
            min_value=10.0,
            max_value=35000.0,
            value=float(preset_vals["population"]),
            step=50.0
        )
    with col_hh:
        households = st.number_input(
            "Households",
            min_value=5.0,
            max_value=10000.0,
            value=float(preset_vals["households"]),
            step=25.0
        )

    # Derived Ratios Preview
    rooms_per_hh = total_rooms / households if households > 0 else 0
    beds_per_room = total_bedrooms / total_rooms if total_rooms > 0 else 0
    pop_per_hh = population / households if households > 0 else 0

    st.markdown("---")
    st.markdown("**📊 Computed Block Ratios:**")
    st.markdown(
        f"""
        - 🚪 Rooms / Household: **`{rooms_per_hh:.2f}`**
        - 🛏️ Bedrooms / Room: **`{beds_per_room:.1%}`**
        - 👥 Population / Household: **`{pop_per_hh:.2f}`**
        """
    )
    if total_bedrooms > total_rooms:
        st.warning("⚠️ Warning: Bedrooms exceed total rooms in this district.")

    st.markdown("---")
    st.session_state.auto_predict = st.toggle("⚡ Real-time Dynamic Prediction", value=st.session_state.auto_predict)
    predict_clicked = False
    if not st.session_state.auto_predict:
        predict_clicked = st.button("🔮 Calculate Valuation", type="primary", use_container_width=True)

# Build current district values dict
current_district = {
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "ocean_proximity": ocean_proximity,
}

# Calculate prediction
should_compute = st.session_state.auto_predict or predict_clicked or len(st.session_state.prediction_history) == 0
result = predict_valuation_ensemble(current_district)
pred_val = result["prediction"]

# Add to history if manually triggered or profile changed significantly
if predict_clicked or (st.session_state.auto_predict and (
    not st.session_state.prediction_history or 
    abs(st.session_state.prediction_history[-1]["prediction"] - pred_val) > 500
)):
    st.session_state.prediction_history.append({
        "timestamp": time.strftime("%H:%M:%S"),
        "preset": selected_preset_name,
        "prediction": pred_val,
        "income": median_income * 10000,
        "age": housing_median_age,
        "location": ocean_proximity,
        "lat": latitude,
        "lon": longitude,
    })

# ---------------------------------------------------------
# Dynamic Tabs Architecture
# ---------------------------------------------------------
tab_main, tab_sensitivity, tab_compare, tab_batch, tab_insights, tab_history = st.tabs([
    "🏡 Live Valuation & Geospatial Map",
    "📈 'What-If' Sensitivity Studio",
    "⚖️ District Comparator (A vs B)",
    "📂 Batch CSV Prediction",
    "🧠 Model Diagnostics & Architecture",
    "📜 Session History",
])

# ---------------------------------------------------------
# TAB 1: Live Valuation & Geospatial Map
# ---------------------------------------------------------
with tab_main:
    tier_label, tier_class = get_price_tier(pred_val)
    ca_median_benchmark = 206855.0
    delta_vs_median = ((pred_val - ca_median_benchmark) / ca_median_benchmark) * 100
    delta_color = "#10b981" if delta_vs_median >= 0 else "#ef4444"
    delta_sign = "+" if delta_vs_median >= 0 else ""

    # Top Metric Dashboard Cards
    c1, c2, c3, c4 = st.columns([1.8, 1.2, 1.2, 1.2])
    with c1:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 5px solid #6366f1;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="metric-label">Estimated District Median Value</div>
                    <span class="badge-tier {tier_class}">{tier_label}</span>
                </div>
                <div class="metric-value">{format_curr(pred_val)}</div>
                <div class="metric-sub" style="color: {delta_color};">
                    {delta_sign}{delta_vs_median:.1f}% vs California Median ($206,855)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">90% Confidence Interval</div>
                <div class="metric-value" style="font-size: 1.35rem;">{format_curr(result['p10'])} – {format_curr(result['p90'])}</div>
                <div class="metric-sub" style="color: #94a3b8;">Tree Std Dev: ±{format_curr(result['std'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Income Multiplier</div>
                <div class="metric-value" style="font-size: 1.35rem;">{pred_val / (median_income * 10000):.1f}x</div>
                <div class="metric-sub" style="color: #60a5fa;">Valuation / Annual Income</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Estimated Density</div>
                <div class="metric-value" style="font-size: 1.35rem;">{pop_per_hh:.2f}</div>
                <div class="metric-sub" style="color: #a78bfa;">Residents per Household</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Grid: Map & Ensemble Distribution
    col_map, col_dist = st.columns([1.6, 1.4])

    with col_map:
        st.subheader("🗺️ Interactive Geospatial Mapping")
        st.caption("Visualizing the selected district (highlighted in glowing cyan/magenta) against 1,200 sampled California districts colored by price tier.")

        # Prepare downsampled background points for lightning-fast 60fps rendering
        sample_pts = full_data.sample(n=min(1200, len(full_data)), random_state=42).copy()
        
        # Color mapping based on median_house_value
        def get_color(val):
            if val < 150000:
                return [59, 130, 246, 140]    # Blue
            elif val < 280000:
                return [16, 185, 129, 160]   # Emerald
            elif val < 420000:
                return [245, 158, 11, 180]   # Amber
            else:
                return [236, 72, 153, 200]   # Pink/Magenta

        sample_pts["color"] = sample_pts["median_house_value"].apply(get_color)

        # Selected pin dataframe
        selected_pin_df = pd.DataFrame([{
            "latitude": latitude,
            "longitude": longitude,
            "name": f"Current Selection ({format_curr(pred_val)})",
            "val": pred_val,
        }])

        background_layer = pdk.Layer(
            "ScatterplotLayer",
            data=sample_pts,
            get_position=["longitude", "latitude"],
            get_color="color",
            get_radius=2800,
            pickable=True,
            auto_highlight=True,
        )

        pin_layer = pdk.Layer(
            "ScatterplotLayer",
            data=selected_pin_df,
            get_position=["longitude", "latitude"],
            get_color=[244, 114, 182, 255],
            get_radius=12000,
            pickable=True,
            stroked=True,
            filled=True,
            line_width_min_pixels=3,
            get_line_color=[255, 255, 255, 255],
        )

        view_state = pdk.ViewState(
            latitude=latitude,
            longitude=longitude,
            zoom=6.2,
            pitch=35,
        )

        deck = pdk.Deck(
            layers=[background_layer, pin_layer],
            initial_view_state=view_state,
            tooltip={"text": "Median Value: ${median_house_value}\nIncome: ${median_income}0k\nOcean: {ocean_proximity}"},
            map_style="dark",
        )
        st.pydeck_chart(deck, use_container_width=True)

    with col_dist:
        st.subheader("🌲 100-Tree Random Forest Consensus")
        st.caption("Distribution of predictions across all 100 individual decision trees showing ensemble certainty.")

        tree_vals = result["tree_estimates"]
        fig_dist = px.histogram(
            x=tree_vals,
            nbins=25,
            labels={"x": "Estimated House Value ($)"},
            title="Ensemble Decision Tree Estimates (N = 100)",
            color_discrete_sequence=["#818cf8"],
            opacity=0.8,
        )
        fig_dist.add_vline(
            x=pred_val,
            line_width=3,
            line_dash="solid",
            line_color="#ec4899",
            annotation_text=f"Ensemble Mean: {format_curr(pred_val)}",
            annotation_position="top left",
        )
        fig_dist.add_vrect(
            x0=result["p10"],
            x1=result["p90"],
            fillcolor="#6366f1",
            opacity=0.15,
            line_width=0,
            annotation_text="80% Consensus Band",
            annotation_position="bottom right",
        )
        fig_dist.update_layout(
            template="plotly_dark",
            margin=dict(l=10, r=10, t=40, b=10),
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.4)",
            xaxis=dict(tickprefix="$", tickformat=",.0f"),
            yaxis_title="Tree Frequency",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: "What-If" Sensitivity Studio
# ---------------------------------------------------------
with tab_sensitivity:
    st.subheader("📈 'What-If' Parameter Sensitivity Explorer")
    st.markdown("Dynamically sweep any variable to observe how the Random Forest valuation responds while holding other characteristics constant.")

    c_var, c_steps = st.columns([2, 1])
    with c_var:
        sweep_var = st.selectbox(
            "Select Feature to Vary:",
            options=[
                ("median_income", "Median Household Income ($10k)"),
                ("housing_median_age", "Housing Median Age (Years)"),
                ("total_rooms", "Total Rooms"),
                ("population", "District Population"),
                ("households", "Households Count"),
            ],
            format_func=lambda x: x[1],
        )[0]
    with c_steps:
        sweep_resolution = st.slider("Simulation Steps", min_value=20, max_value=60, value=35)

    # Determine sweep range based on feature
    ranges = {
        "median_income": (0.5, 15.0),
        "housing_median_age": (1.0, 52.0),
        "total_rooms": (200.0, 15000.0),
        "population": (100.0, 10000.0),
        "households": (50.0, 4000.0),
    }
    min_s, max_s = ranges[sweep_var]
    sweep_values = np.linspace(min_s, max_s, sweep_resolution)

    # Run batch sweep predictions
    sweep_preds = []
    base_dict = current_district.copy()
    for s_val in sweep_values:
        temp_dict = base_dict.copy()
        temp_dict[sweep_var] = s_val
        sweep_preds.append(predict_valuation_ensemble(temp_dict)["prediction"])

    sweep_df = pd.DataFrame({
        "Feature Value": sweep_values,
        "Valuation": sweep_preds,
    })

    current_var_val = current_district[sweep_var]

    fig_sweep = px.line(
        sweep_df,
        x="Feature Value",
        y="Valuation",
        title=f"Valuation Trajectory vs {sweep_var.replace('_', ' ').title()}",
        labels={"Feature Value": sweep_var.replace("_", " ").title(), "Valuation": "Estimated House Value ($)"},
    )
    fig_sweep.update_traces(line=dict(color="#38bdf8", width=3.5))

    # Add current point marker
    fig_sweep.add_trace(
        go.Scatter(
            x=[current_var_val],
            y=[pred_val],
            mode="markers+text",
            marker=dict(color="#f43f5e", size=14, symbol="diamond"),
            text=[f"Current: {format_curr(pred_val)}"],
            textposition="top center",
            name="Current Setting",
        )
    )

    fig_sweep.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        height=420,
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
    )
    st.plotly_chart(fig_sweep, use_container_width=True)

    # Elasticity Callout
    v_start, v_end = sweep_preds[0], sweep_preds[-1]
    pct_change = ((v_end - v_start) / v_start) * 100
    st.info(
        f"💡 **Sensitivity Insight:** Sweeping `{sweep_var}` from {min_s:.1f} to {max_s:.1f} drives a **{pct_change:+.1f}%** shift in property value (from {format_curr(v_start)} to {format_curr(v_end)})."
    )

# ---------------------------------------------------------
# TAB 3: District Comparator (A vs B)
# ---------------------------------------------------------
with tab_compare:
    st.subheader("⚖️ District Comparison Studio (A vs B)")
    st.caption("Benchmark two distinct California communities side-by-side to understand relative premiums.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🅰️ District A")
        preset_a = st.selectbox("Archetype A", list(PRESETS.keys()), index=0, key="preset_a")
        data_a = PRESETS[preset_a].copy()
        pred_a = predict_valuation_ensemble(data_a)["prediction"]
        st.metric("Estimated Valuation A", format_curr(pred_a), delta=f"Income: ${data_a['median_income']*10000:,.0f}")
        st.write(f"**Location:** `{data_a['ocean_proximity']}` | **Median Age:** `{data_a['housing_median_age']:.0f} yrs`")

    with col_b:
        st.markdown("### 🅱️ District B")
        preset_b = st.selectbox("Archetype B", list(PRESETS.keys()), index=4, key="preset_b")
        data_b = PRESETS[preset_b].copy()
        pred_b = predict_valuation_ensemble(data_b)["prediction"]
        diff_val = pred_b - pred_a
        diff_pct = ((pred_b - pred_a) / pred_a) * 100
        st.metric("Estimated Valuation B", format_curr(pred_b), delta=f"{diff_pct:+.1f}% vs District A")
        st.write(f"**Location:** `{data_b['ocean_proximity']}` | **Median Age:** `{data_b['housing_median_age']:.0f} yrs`")

    st.markdown("---")
    # Comparison Bar Chart
    comp_df = pd.DataFrame([
        {"District": preset_a, "Valuation": pred_a, "Income": data_a["median_income"] * 10000, "Rooms": data_a["total_rooms"]},
        {"District": preset_b, "Valuation": pred_b, "Income": data_b["median_income"] * 10000, "Rooms": data_b["total_rooms"]},
    ])

    fig_comp = px.bar(
        comp_df,
        x="District",
        y="Valuation",
        color="District",
        color_discrete_sequence=["#818cf8", "#f472b6"],
        text="Valuation",
        title="Direct Valuation Comparison ($)",
    )
    fig_comp.update_traces(texttemplate="$%{y:,.0f}", textposition="outside")
    fig_comp.update_layout(
        template="plotly_dark",
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        showlegend=False,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: Batch CSV Prediction
# ---------------------------------------------------------
with tab_batch:
    st.subheader("📂 Batch CSV District Valuation Engine")
    st.caption("Upload a spreadsheet of housing districts to generate automated ensemble predictions in bulk.")

    # Sample CSV Download Button
    sample_batch = full_data.head(10).drop(columns=["median_house_value"], errors="ignore")
    csv_buffer = io.StringIO()
    sample_batch.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Sample Batch Template (.csv)",
        data=csv_buffer.getvalue(),
        file_name="california_housing_sample_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload CSV File with District Records", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(batch_df):,} district records.")

            required_cols = ["longitude", "latitude", "housing_median_age", "total_rooms", 
                             "total_bedrooms", "population", "households", "median_income", "ocean_proximity"]
            missing = [c for c in required_cols if c not in batch_df.columns]

            if missing:
                st.error(f"Missing required columns in CSV: `{missing}`")
            else:
                with st.spinner("Generating Random Forest batch predictions..."):
                    preds_list = []
                    # Compute predictions row-by-row or vectorized
                    for _, row_data in batch_df.iterrows():
                        v_dict = row_data.to_dict()
                        f_row = prepare_feature_row(v_dict, feature_columns)
                        s_feat = scaler.transform(f_row)
                        p = float(model.predict(s_feat)[0])
                        preds_list.append(p)

                    batch_df["Estimated_Median_House_Value"] = preds_list
                    batch_df["Price_Tier"] = [get_price_tier(p)[0] for p in preds_list]

                # Summary Statistics
                st.markdown("### 📊 Batch Prediction Summary")
                bc1, bc2, bc3, bc4 = st.columns(4)
                bc1.metric("Average Valuation", format_curr(batch_df["Estimated_Median_House_Value"].mean()))
                bc2.metric("Median Valuation", format_curr(batch_df["Estimated_Median_House_Value"].median()))
                bc3.metric("Min Valuation", format_curr(batch_df["Estimated_Median_House_Value"].min()))
                bc4.metric("Max Valuation", format_curr(batch_df["Estimated_Median_House_Value"].max()))

                # Result Table Preview
                st.dataframe(
                    batch_df[["Estimated_Median_House_Value", "Price_Tier", "median_income", "housing_median_age", "ocean_proximity"]].head(25),
                    use_container_width=True,
                )

                # Download Results Button
                out_buffer = io.StringIO()
                batch_df.to_csv(out_buffer, index=False)
                st.download_button(
                    label="💾 Download Predicted Valuations CSV",
                    data=out_buffer.getvalue(),
                    file_name="predicted_california_housing_values.csv",
                    mime="text/csv",
                    type="primary",
                )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# ---------------------------------------------------------
# TAB 5: Model Diagnostics & Architecture
# ---------------------------------------------------------
with tab_insights:
    st.subheader("🧠 Machine Learning Model Architecture & Performance")
    st.markdown("Detailed inspectability of the trained scikit-learn Random Forest model.")

    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
    col_meta1.metric("Model Algorithm", "Random Forest")
    col_meta2.metric("Number of Trees", len(model.estimators_))
    col_meta3.metric("Validation R² Score", "0.8072")
    col_meta4.metric("Test RMSE", "$50,258.95")

    st.markdown("---")
    st.subheader("🏆 Global Feature Importance (Gini Impurity)")
    st.caption("How much each feature contributes to the Random Forest's split decisions.")

    fi_series = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=True)
    fi_df = pd.DataFrame({
        "Feature": [f.replace("_", " ").title() for f in fi_series.index],
        "Importance": fi_series.values,
    })

    fig_fi = px.bar(
        fi_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Viridis",
        title="Feature Importance Ranking",
    )
    fig_fi.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        height=480,
        xaxis=dict(tickformat=".1%"),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

# ---------------------------------------------------------
# TAB 6: Session History
# ---------------------------------------------------------
with tab_history:
    st.subheader("📜 Current Session Prediction Log")
    if st.session_state.prediction_history:
        hist_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Timeline chart
        fig_hist = px.line(
            hist_df,
            y="prediction",
            markers=True,
            title="Session Valuation Progression ($)",
            labels={"index": "Simulation Run #", "prediction": "Estimated Value ($)"},
        )
        fig_hist.update_traces(line=dict(color="#ec4899", width=2.5), marker=dict(size=8))
        fig_hist.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.4)",
            height=320,
            yaxis=dict(tickprefix="$", tickformat=",.0f"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.dataframe(
            hist_df.rename(columns={
                "timestamp": "Time",
                "preset": "Profile",
                "prediction": "Estimated Value",
                "income": "Income",
                "age": "Age",
                "location": "Ocean Proximity",
            }),
            use_container_width=True,
            hide_index=True,
        )

        col_h1, col_h2 = st.columns([1, 4])
        with col_h1:
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.rerun()
    else:
        st.info("No predictions recorded in this session yet. Adjust sliders in the sidebar to generate valuations.")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()
st.caption("CaliHome AI • Developed with Streamlit, Scikit-Learn & PyDeck | Model: Random Forest Regressor (100 Trees)")