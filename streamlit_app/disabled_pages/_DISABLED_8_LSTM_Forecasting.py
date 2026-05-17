import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

sys.path.append(PROJECT_ROOT)

import streamlit as st

from src.dl.lstm.predict import forecast_lstm


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(

    page_title="LSTM Forecasting",

    layout="wide"

)

# =========================================================
# TITLE
# =========================================================

st.title("📈 Deep Learning LSTM Forecasting")

st.write(
    "Forecast future ecommerce orders using LSTM."
)

# =========================================================
# INPUTS
# =========================================================

sequence = []

st.subheader("Enter Last 10 Sales Values")

for i in range(10):

    value = st.number_input(

        f"Day {i+1}",

        value=float(i+1)

    )

    sequence.append(value)

# =========================================================
# BUTTON
# =========================================================

if st.button("Forecast"):

    result = forecast_lstm(sequence)

    st.subheader("Forecast Result")

    st.success(

        result['forecasted_orders']

    )