"""
Recommendation page — item-based collaborative filtering.
Finds top-5 products most similar to a given product name.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


@st.cache_resource
def load_similarity_matrix():
    """Load the pre-computed cosine similarity matrix."""
    return pd.read_pickle(MODELS_DIR / "similarity_matrix.pkl")


def find_product(query: str, product_index: pd.Index) -> str | None:
    """
    Match a user query to a product name (case-insensitive).
    Falls back to partial substring match if exact match fails.
    """
    query_upper = query.upper().strip()
    # Exact case-insensitive match
    matches = [p for p in product_index if p.upper() == query_upper]
    if matches:
        return matches[0]
    # Partial substring match
    matches = [p for p in product_index if query_upper in p.upper()]
    return matches[0] if matches else None


def get_recommendations(product_name: str, sim_df: pd.DataFrame, top_n: int = 5) -> list[tuple[str, float]]:
    """Return top-N most similar products with cosine similarity scores."""
    matched = find_product(product_name, sim_df.index)
    if matched is None:
        return []
    similar = (
        sim_df[matched]
        .sort_values(ascending=False)
        .drop(matched)
        .head(top_n)
    )
    return list(zip(similar.index, similar.values))


st.title("🔍 Product Recommender")
st.markdown(
    "Enter a product name to find the **top 5 most similar products** "
    "based on **item-based collaborative filtering** (cosine similarity "
    "on customer purchase patterns)."
)

sim_path = MODELS_DIR / "similarity_matrix.pkl"
if not sim_path.exists():
    st.error(
        "Similarity matrix not found. "
        "Run `python train_spectrum_model.py` first."
    )
    st.stop()

sim_df = load_similarity_matrix()

# Show sample products to help the user
with st.expander("Browse sample product names"):
    sample = sorted(sim_df.index.tolist())[:30]
    st.write(", ".join(sample))
    st.caption(f"Showing 30 of {len(sim_df):,} products. Type any name or partial match.")

product_input = st.text_input(
    "Enter Product Name",
    placeholder="e.g. GREEN VINTAGE SPOT BEAKER",
)

if st.button("Recommend", type="primary") and product_input.strip():
    matched = find_product(product_input, sim_df.index)

    if matched is None:
        st.warning(
            f"No product found matching '{product_input}'. "
            "Try a different name or a partial match from the sample list above."
        )
    else:
        if matched.upper() != product_input.upper().strip():
            st.caption(f"Matched to: **{matched}**")

        recs = get_recommendations(matched, sim_df, top_n=5)

        if not recs:
            st.warning("No recommendations available for this product.")
        else:
            st.subheader("Recommended Products")
            for i, (prod, score) in enumerate(recs, 1):
                st.markdown(
                    f"**{i}.** {prod} &nbsp; — &nbsp; "
                    f"<span style='color:gray;'>similarity: {score:.3f}</span>",
                    unsafe_allow_html=True,
                )
