import sys
import os


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

sys.path.append(PROJECT_ROOT)


# =========================================================
# IMPORTS
# =========================================================

import streamlit as st

from src.ml.recommendations.predict import (
    recommend_products
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Recommendation System",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🛍 Product Recommendation System")

st.write(
    "Hybrid Product Recommendation Engine"
)


# =========================================================
# INPUT
# =========================================================

product_id = st.text_input(
    "Enter Product ID"
)


# =========================================================
# BUTTON
# =========================================================

if st.button("Get Recommendations"):

    result = recommend_products(product_id)

    st.subheader("Recommended Products")

    for product in result[
        'recommended_products'
    ]:

        st.success(product)