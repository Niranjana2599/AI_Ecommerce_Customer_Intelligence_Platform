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

from src.ml.delay_prediction.predict import predict_delay


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Delivery Delay Prediction",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🚚 Delivery Delay Prediction")

st.write(
    "Predict estimated delivery delay using CatBoost ML."
)

# =========================================================
# INPUTS
# =========================================================

col1, col2 = st.columns(2)

with col1:

    customer_age = st.number_input(
        "Customer Age",
        value=30
    )

    delivery_duration_days = st.number_input(
        "Delivery Duration Days",
        value=5.0
    )

    approval_time_hours = st.number_input(
        "Approval Time Hours",
        value=2.0
    )

    freight_value = st.number_input(
        "Freight Value",
        value=15.0
    )

    payment_value = st.number_input(
        "Payment Value",
        value=135.0
    )

with col2:

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

    frequency = st.number_input(
        "Customer Frequency",
        value=5
    )

    monetary = st.number_input(
        "Customer Monetary",
        value=500
    )

# =========================================================
# PREDICT BUTTON
# =========================================================

if st.button("Predict Delay"):

    input_data = {

        "order_status": "delivered",

        "delivery_duration_days": delivery_duration_days,
        "approval_time_hours": approval_time_hours,

        "purchase_month": 5,
        "purchase_year": 2018,
        "purchase_weekday": 2,

        "customer_gender": "Female",
        "customer_age": customer_age,
        "customer_zip_code_prefix": 12345,
        "customer_city": "sao paulo",
        "customer_state": "SP",
        "customer_segment": 1,

        "order_item_id": 1,

        "price_x": 120,
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
        "review_word_count": review_word_count,

        "total_orders": 5,
        "total_spent": 500,

        "avg_order_value": 100,
        "avg_review_score": 4.2,

        "avg_delivery_delay": 1,

        "unique_products_purchased": 3,

        "Recency": 30,
        "Frequency": frequency,
        "Monetary": monetary
    }

    result = predict_delay(input_data)

    predicted_delay = result["predicted_delay_days"]

    st.subheader("Prediction Result")

    st.success(
        f"🚚 Predicted Delivery Delay: {predicted_delay} days"
    )