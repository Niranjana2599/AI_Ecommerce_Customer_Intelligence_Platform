import sys
import os

# =========================================================
# ADD PROJECT ROOT TO PYTHON PATH
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

sys.path.append(PROJECT_ROOT)

# =========================================================
# IMPORTS
# =========================================================

import streamlit as st

from src.ml.churn.predict import predict_churn


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Churn Prediction",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("📉 Customer Churn Prediction")

st.write(
    "Predict whether a customer is likely to churn."
)

# =========================================================
# CUSTOMER DETAILS
# =========================================================

st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:

    order_status = st.selectbox(
        "Order Status",
        ["delivered", "shipped", "processing"]
    )

    delivery_duration_days = st.number_input(
        "Delivery Duration Days",
        value=5.0
    )

    delivery_delay_days = st.number_input(
        "Delivery Delay Days",
        value=1.0
    )

    approval_time_hours = st.number_input(
        "Approval Time Hours",
        value=2.0
    )

    purchase_month = st.number_input(
        "Purchase Month",
        value=5
    )

    purchase_year = st.number_input(
        "Purchase Year",
        value=2018
    )

    customer_gender = st.selectbox(
        "Customer Gender",
        ["Male", "Female"]
    )

    customer_age = st.number_input(
        "Customer Age",
        value=30
    )

    customer_city = st.text_input(
        "Customer City",
        value="sao paulo"
    )

    customer_state = st.text_input(
        "Customer State",
        value="SP"
    )

with col2:

    price_x = st.number_input(
        "Product Price",
        value=120.0
    )

    freight_value = st.number_input(
        "Freight Value",
        value=15.0
    )

    payment_value = st.number_input(
        "Payment Value",
        value=135.0
    )

    review_score = st.slider(
        "Review Score",
        1,
        5,
        4
    )

    review_length = st.number_input(
        "Review Length",
        value=50
    )

    review_word_count = st.number_input(
        "Review Word Count",
        value=10
    )

# =========================================================
# PREDICT BUTTON
# =========================================================

if st.button("Predict Churn"):

    input_data = {

        "order_status": order_status,

        "delivery_duration_days": delivery_duration_days,
        "delivery_delay_days": delivery_delay_days,
        "approval_time_hours": approval_time_hours,

        "purchase_month": purchase_month,
        "purchase_year": purchase_year,
        "purchase_weekday": 2,

        "customer_gender": customer_gender,
        "customer_age": customer_age,
        "customer_zip_code_prefix": 12345,
        "customer_city": customer_city,
        "customer_state": customer_state,
        "customer_segment": 1,

        "order_item_id": 1,

        "price_x": price_x,
        "freight_value": freight_value,
        "discount_rate": 0.1,

        "product_category_name": 5,
        "product_name": "product_a",
        "product_brand": 2,

        "product_weight_g": 500,
        "product_length_cm": 20,
        "product_height_cm": 10,
        "product_width_cm": 15,

        "cost": 80,
        "price_y": 120,

        "seller_contact_gender": "Male",
        "seller_contact_age": 40,
        "seller_zip_code_prefix": 22222,
        "seller_city": "rio",
        "seller_state": "RJ",

        "payment_sequential": 1,
        "payment_type": 2,
        "payment_installments": 2,
        "payment_value": payment_value,

        "review_score": review_score,

        "purchase_day": 10,
        "purchase_hour": 15,
        "purchase_weekend": 0,

        "payment_apple_pay": 0,
        "payment_bank_transfer": 0,
        "payment_boleto": 0,
        "payment_credit_card": 1,
        "payment_debit_card": 0,
        "payment_paypal": 0,
        "payment_voucher": 0,

        "payment_apple_pay.1": 0,
        "payment_bank_transfer.1": 0,
        "payment_boleto.1": 0,
        "payment_credit_card.1": 1,
        "payment_debit_card.1": 0,
        "payment_paypal.1": 0,
        "payment_voucher.1": 0,

        "review_length": review_length,
        "review_word_count": review_word_count
    }

    result = predict_churn(input_data)

    prediction = result["prediction"]

    probability = result["churn_probability"]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            f"⚠ Customer likely to churn\n\nProbability: {probability}"
        )

    else:

        st.success(
            f"✅ Customer likely to stay\n\nProbability: {probability}"
        )