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

from src.nlp.sentiment_analysis import (
    predict_sentiment
)

from src.nlp.summarization import (
    summarize_review
)

from src.nlp.complaint_detection import (
    detect_complaint
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="NLP Sentiment Analysis",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🧠 NLP Review Intelligence System")

st.write(
    "Analyze customer reviews using NLP."
)

# =========================================================
# INPUT
# =========================================================

review_text = st.text_area(
    "Enter Customer Review"
)

# =========================================================
# BUTTON
# =========================================================

if st.button("Analyze Review"):

    # =====================================================
    # SENTIMENT
    # =====================================================

    sentiment_result = predict_sentiment(
        review_text
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    summary_result = summarize_review(
        review_text
    )

    # =====================================================
    # COMPLAINT
    # =====================================================

    complaint_result = detect_complaint(
        review_text
    )

    # =====================================================
    # OUTPUTS
    # =====================================================

    st.subheader("Sentiment Prediction")

    st.write(sentiment_result)

    st.subheader("Complaint Detection")

    st.write(complaint_result)

    st.subheader("Review Summary")

    st.success(summary_result)