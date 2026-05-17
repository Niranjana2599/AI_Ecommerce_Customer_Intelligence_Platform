import numpy as np
import joblib
import tensorflow as tf

from src.utils.path_config import MODELS_DIR

# =========================================================
# LOAD MODEL
# =========================================================

model = tf.keras.models.load_model(
    "artifacts/models/lstm_forecasting_model.h5",
    compile=False
)

scaler = joblib.load(
    MODELS_DIR / "lstm_scaler.pkl"
)


# =========================================================
# FORECAST
# =========================================================

def forecast_lstm(input_sequence):

    input_sequence = np.array(
        input_sequence
    )

    input_sequence = input_sequence.reshape(
        1,
        len(input_sequence),
        1
    )

    prediction = model.predict(
        input_sequence
    )

    prediction = scaler.inverse_transform(
        prediction
    )

    return {

        "forecasted_orders":

            round(float(prediction[0][0]), 2)

    }


# =========================================================
# TESTING
# =========================================================

if __name__ == "__main__":

    sample_sequence = [

        0.1,
        0.2,
        0.15,
        0.3,
        0.4,
        0.35,
        0.5,
        0.55,
        0.6,
        0.65

    ]

    result = forecast_lstm(
        sample_sequence
    )

    print(result)