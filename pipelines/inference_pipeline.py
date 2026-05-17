# =========================================================
# INFERENCE PIPELINE
# =========================================================

from src.ml.churn.predict import predict_churn


# =========================================================
# RUN INFERENCE
# =========================================================


def run_inference(data):

    result = predict_churn(data)

    return result