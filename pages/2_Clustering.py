"""
Clustering page — predict customer spectrum from RFM inputs.
Uses the saved StandardScaler and KMeans model from models/.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Colour mapping for segment display
COLOR_MAP = {
    "High-Value": "#2ECC71",
    "Regular": "#3498DB",
    "Occasional": "#F39C12",
    "At-Risk": "#E74C3C",
}

SEGMENT_DESCRIPTIONS = {
    "High-Value": "Best customers — frequent, recent, big spenders. Reward & retain.",
    "Regular": "Steady buyers with upsell potential via targeted promotions.",
    "Occasional": "Infrequent buyers who need re-engagement campaigns.",
    "At-Risk": "Haven't bought recently. Win-back campaigns and discounts recommended.",
}


@st.cache_resource
def load_clustering_models():
    """Load persisted scaler, KMeans model, and cluster label mapping."""
    scaler = joblib.load(MODELS_DIR / "rfm_scaler.pkl")
    kmeans = joblib.load(MODELS_DIR / "kmeans_model.pkl")
    with open(MODELS_DIR / "cluster_labels.json", encoding="utf-8") as f:
        labels = json.load(f)
    return scaler, kmeans, labels


def predict_segment(recency: float, frequency: float, monetary: float, scaler, kmeans, cluster_labels: dict) -> str:
    """Scale RFM input and predict the customer spectrum segment."""
    x = np.array([[recency, frequency, monetary]])
    x_scaled = scaler.transform(x)
    cluster_id = str(kmeans.predict(x_scaled)[0])
    return cluster_labels[cluster_id]


st.title("🎯 Customer Segmentation")
st.markdown(
    "Enter a customer's **RFM metrics** below to predict which "
    "**Shopper Spectrum** segment they belong to."
)

# Check models exist before loading
required_files = ["rfm_scaler.pkl", "kmeans_model.pkl", "cluster_labels.json"]
missing = [f for f in required_files if not (MODELS_DIR / f).exists()]

if missing:
    st.error(
        f"Missing model files: {', '.join(missing)}. "
        "Run `python train_spectrum_model.py` first."
    )
    st.stop()

scaler, kmeans, cluster_labels = load_clustering_models()

# ─── Input form ──────────────────────────────────────────────────────────────
st.subheader("Customer RFM Input")
col1, col2, col3 = st.columns(3)

with col1:
    recency = st.number_input(
        "Recency (days since last purchase)",
        min_value=0, max_value=1000, value=100, step=1,
        help="Lower = more recent purchase (better)",
    )
with col2:
    frequency = st.number_input(
        "Frequency (number of unique orders)",
        min_value=1, max_value=500, value=5, step=1,
        help="Total distinct invoices placed by the customer",
    )
with col3:
    monetary = st.number_input(
        "Monetary (total spend £)",
        min_value=0.0, max_value=1_000_000.0, value=500.0, step=10.0,
        help="Total revenue from this customer",
    )

if st.button("Predict Segment", type="primary"):
    segment = predict_segment(recency, frequency, monetary, scaler, kmeans, cluster_labels)
    color = COLOR_MAP.get(segment, "gray")

    st.markdown("---")
    st.markdown(
        f"## This customer belongs to: "
        f"<span style='color:{color};font-size:1.4em;'><b>{segment}</b></span>",
        unsafe_allow_html=True,
    )
    st.info(SEGMENT_DESCRIPTIONS.get(segment, ""))

    # Show scaled features for transparency
    x_scaled = scaler.transform(np.array([[recency, frequency, monetary]]))
    st.caption(
        f"Scaled features → R: {x_scaled[0][0]:.3f}, "
        f"F: {x_scaled[0][1]:.3f}, M: {x_scaled[0][2]:.3f}"
    )

# ─── Reference guide ─────────────────────────────────────────────────────────
with st.expander("Segment Reference Guide"):
    st.markdown(
        """
| Segment | Recency | Frequency | Monetary | Action |
|---------|---------|-----------|----------|--------|
| High-Value | Low (recent) | High | High | Loyalty rewards, VIP offers |
| Regular | Medium | Medium | Medium | Upsell, cross-sell |
| Occasional | Medium-High | Low | Low | Re-engagement emails |
| At-Risk | High (old) | Low | Low-Med | Win-back discounts |
        """
    )
