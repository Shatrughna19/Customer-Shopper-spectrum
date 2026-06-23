"""
Home page — project overview and dataset statistics.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

st.title("🏠 Shopper Spectrum — Home")
st.markdown(
    """
### Project Overview
**Shopper Spectrum** segments online retail customers into four behavioral
spectrums using **RFM analysis** (Recency, Frequency, Monetary) and
**KMeans clustering**, then recommends similar products via
**item-based collaborative filtering**.

| Segment | Profile |
|---------|---------|
| **High-Value** | Recent, frequent, high-spending customers |
| **Regular** | Steady mid-range buyers |
| **Occasional** | Infrequent, low-spend purchasers |
| **At-Risk** | Haven't purchased recently — win-back targets |
"""
)

st.markdown("---")
st.subheader("📊 Dataset Statistics")

# Load cleaned data for display stats
cleaned_path = DATA_DIR / "online_retail_cleaned.csv"
rfm_path = DATA_DIR / "rfm_features.csv"

if cleaned_path.exists() and rfm_path.exists():
    df = pd.read_csv(cleaned_path)
    rfm = pd.read_csv(rfm_path)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions", f"{len(df):,}")
    col2.metric("Unique Customers", f"{df['CustomerID'].nunique():,}")
    col3.metric("Unique Products", f"{df['Description'].nunique():,}")
    col4.metric("Countries", f"{df['Country'].nunique():,}")

    col5, col6, col7 = st.columns(3)
    col5.metric("Avg Customer Spend (£)", f"{rfm['Monetary'].mean():,.2f}")
    col6.metric("Avg Order Frequency", f"{rfm['Frequency'].mean():.1f}")
    col7.metric("Avg Recency (days)", f"{rfm['Recency'].mean():.0f}")

    # Show segment distribution if training has been run
    rfm_seg_path = DATA_DIR / "rfm_with_segments.csv"
    if rfm_seg_path.exists():
        rfm_seg = pd.read_csv(rfm_seg_path)
        st.subheader("Customer Segment Distribution")
        seg_counts = rfm_seg["Segment"].value_counts()
        st.bar_chart(seg_counts)
else:
    st.warning(
        "Data files not found. Please run **CustomerEDA.ipynb** first, "
        "then **train_spectrum_model.py** to generate models."
    )

st.markdown("---")
st.markdown(
    """
### How to Use
1. Run `CustomerEDA.ipynb` to clean data and compute RFM features.
2. Run `python train_spectrum_model.py` to train clustering & recommender models.
3. Use **Clustering** page to predict segments for new customers.
4. Use **Recommendation** page to find similar products.
"""
)
