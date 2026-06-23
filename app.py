"""
Shopper Spectrum — Streamlit Application
========================================
Main entry point for the customer segmentation GUI and product
recommendation engine. Run from the final/ directory:

    streamlit run app.py
"""

import streamlit as st

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar navigation ──────────────────────────────────────────────────────
st.sidebar.title("🛒 Shopper Spectrum")
st.sidebar.markdown(
    "Customer segmentation & product recommendations powered by "
    "KMeans RFM clustering and item-based collaborative filtering."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Navigate** using the pages in the sidebar:")
st.sidebar.markdown("- **Home** — project overview & dataset stats")
st.sidebar.markdown("- **Clustering** — predict customer spectrum")
st.sidebar.markdown("- **Recommendation** — similar product finder")

# ─── Landing content (shown on the root page before pages/ home loads) ───────
st.title("Shopper Spectrum")
st.markdown(
    """
Welcome to the **Shopper Spectrum** application.

Use the sidebar to navigate between modules:

| Page | Description |
|------|-------------|
| **Home** | Dataset statistics and project overview |
| **Clustering** | Enter RFM values to predict a customer's spectrum segment |
| **Recommendation** | Enter a product name to get top-5 similar products |

> **Note:** Ensure `train_spectrum_model.py` has been run so that
> trained models exist in the `models/` folder.
"""
)
