import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

sys.path.append(PROJECT_ROOT)

import streamlit as st

from src.dl.transformer.predict import (
    predict_sentiment
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(

    page_title="BERT Sentiment Analysis",

    layout="wide"

)

# =========================================================
# TITLE
# =========================================================

st.title("🤖 BERT Sentiment Analysis")

st.write(
    "Transformer-based NLP sentiment analysis."
)

# =========================================================
# INPUT
# =========================================================

text = st.text_area(
    "Enter Customer Review"
)

# =========================================================
# BUTTON
# =========================================================

if st.button("Analyze Sentiment"):

    result = predict_sentiment(text)

    st.subheader("Prediction")

    st.success(result['sentiment'])