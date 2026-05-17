import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

sys.path.append(PROJECT_ROOT)

import streamlit as st
import pandas as pd

from src.forecasting.predict import (
    forecast_orders
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(

    page_title="Time Series Forecasting",

    layout="wide"

)

# =========================================================
# TITLE
# =========================================================

st.title("📈 Time Series Forecasting Dashboard")

st.write(
    "Forecast future ecommerce orders using Prophet."
)

# =========================================================
# INPUT
# =========================================================

forecast_days = st.slider(

    "Forecast Days",

    7,
    90,
    30

)

# =========================================================
# BUTTON
# =========================================================

if st.button("Generate Forecast"):

    forecast_result = forecast_orders(
        forecast_days
    )

    forecast_df = pd.DataFrame(
        forecast_result
    )

    st.subheader("Forecast Results")

    st.dataframe(forecast_df)

    st.subheader("Forecast Chart")

    st.line_chart(
        forecast_df[['yhat']]
    )