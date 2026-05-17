import pickle
import pandas as pd


# =========================================================
# LOAD MODEL
# =========================================================

from src.utils.path_config import MODELS_DIR

with open(MODELS_DIR / "churn_model.pkl", "rb") as f:
    model = pickle.load(f)
    
with open(MODELS_DIR / "churn_features.pkl", "rb") as f:
    feature_columns = pickle.load(f)

with open(MODELS_DIR / "churn_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)


# =========================================================
# PREDICT FUNCTION
# =========================================================

def predict_churn(input_data):

    # =====================================================
    # CONVERT INPUT TO DATAFRAME
    # =====================================================

    input_df = pd.DataFrame([input_data])

    # =====================================================
    # ENSURE ALL FEATURES EXIST
    # =====================================================

    input_df = input_df.reindex(columns=feature_columns)

    # =====================================================
    # HANDLE CATEGORICAL COLUMNS
    # =====================================================

    for col in input_df.columns:

        # Convert object columns safely
        if input_df[col].dtype == "object":

            if col in encoders:

                try:

                    input_df[col] = encoders[col].transform(
                        input_df[col].astype(str)
                    )

                except:

                    input_df[col] = 0

            else:

                input_df[col] = 0

    # =====================================================
    # FILL NULL VALUES
    # =====================================================

    input_df = input_df.fillna(0)

    # =====================================================
    # CONVERT ALL TO NUMERIC
    # =====================================================

    input_df = input_df.apply(pd.to_numeric)

    # =====================================================
    # PREDICTION
    # =====================================================

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    # =====================================================
    # RETURN RESULT
    # =====================================================

    result = {

        "prediction": int(prediction),

        "churn_probability": round(float(probability), 4)

    }

    return result


# =========================================================
# TESTING
# =========================================================

if __name__ == "__main__":

    sample_data = {

        "order_status": "delivered",

        "delivery_duration_days": 5,
        "delivery_delay_days": 1,
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
        "review_word_count": 10
    }

    result = predict_churn(sample_data)

    print("\nPrediction Result:\n")

    print(result)