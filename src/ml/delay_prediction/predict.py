import pickle
import pandas as pd

from catboost import Pool


# =========================================================
# LOAD MODEL
# =========================================================

from src.utils.path_config import MODELS_DIR

with open(MODELS_DIR / "delay_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(MODELS_DIR / "delay_features.pkl", "rb") as f:
    feature_columns = pickle.load(f)

with open(MODELS_DIR / "delay_cat_cols.pkl", "rb") as f:
    categorical_cols = pickle.load(f)


# =========================================================
# PREDICT DELIVERY DELAY
# =========================================================

def predict_delay(input_data):

    # =====================================================
    # CONVERT INPUT TO DATAFRAME
    # =====================================================

    input_df = pd.DataFrame([input_data])

    # =====================================================
    # ENSURE FEATURE ORDER
    # =====================================================

    input_df = input_df.reindex(columns=feature_columns)

    # =====================================================
    # FILL NULL VALUES
    # =====================================================

    input_df = input_df.fillna(0)

    # =====================================================
    # CONVERT CATEGORICAL COLUMNS TO STRING
    # =====================================================

    for col in categorical_cols:

        if col in input_df.columns:

            input_df[col] = input_df[col].astype(str)

    # =====================================================
    # CONVERT NUMERIC COLUMNS
    # =====================================================

    for col in input_df.columns:

        if col not in categorical_cols:

            input_df[col] = pd.to_numeric(
                input_df[col],
                errors="coerce"
            )

    # =====================================================
    # CREATE CATBOOST POOL
    # =====================================================

    prediction_data = Pool(
        data=input_df,
        cat_features=categorical_cols
    )

    # =====================================================
    # PREDICT
    # =====================================================

    prediction = model.predict(prediction_data)[0]

    # =====================================================
    # RETURN RESULT
    # =====================================================

    result = {

        "predicted_delay_days": round(float(prediction), 2)

    }

    return result


# =========================================================
# TESTING
# =========================================================

if __name__ == "__main__":

    sample_data = {

        "order_status": "delivered",

        "delivery_duration_days": 5,
        "approval_time_hours": 2,

        "purchase_month": 5,
        "purchase_year": 2018,
        "purchase_weekday": 2,

        "customer_gender": "Female",
        "customer_age": 30,
        "customer_zip_code_prefix": 12345,
        "customer_city": "sao paulo",
        "customer_state": "SP",
        "customer_segment": 1,

        "order_item_id": 1,

        "price_x": 120,
        "freight_value": 15,
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
        "payment_value": 135,

        "review_score": 4,

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

        "review_length": 50,
        "review_word_count": 10,

        "total_orders": 5,
        "total_spent": 500,

        "avg_order_value": 100,
        "avg_review_score": 4.2,

        "avg_delivery_delay": 1,

        "unique_products_purchased": 3,

        "Recency": 30,
        "Frequency": 5,
        "Monetary": 500
    }

    result = predict_delay(sample_data)

    print("\nPrediction Result:\n")

    print(result)