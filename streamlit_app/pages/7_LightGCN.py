import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

sys.path.append(PROJECT_ROOT)

import streamlit as st

from src.dl.lightgcn.predict import (
    recommend_products
)

from src.dl.lightgcn.predict import (
    user_encoder
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(

    page_title="LightGCN Recommendations",

    layout="wide"

)

# =========================================================
# TITLE
# =========================================================

st.title("🧠 LightGCN Recommendation System")

st.write(
    "Deep Learning Product Recommendation Engine"
)

# =========================================================
# CUSTOMER INPUT
# =========================================================

customer_id = st.selectbox(

    "Select Customer",

    user_encoder.classes_[:100]

)

top_k = st.slider(

    "Number of Recommendations",

    1,
    20,
    10

)

# =========================================================
# BUTTON
# =========================================================

if st.button("Generate Recommendations"):

    result = recommend_products(

        customer_id,

        top_k

    )

    st.subheader("Recommended Products")

    st.write(

        result['recommended_products']

    )