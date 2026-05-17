import numpy as np

from sklearn.metrics import (

    mean_absolute_error,

    mean_squared_error,

    r2_score

)


# =========================================================
# EVALUATE FORECAST
# =========================================================

def evaluate_forecast(

    actual,
    predicted

):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    print("\n=================================================")
    print("LSTM FORECAST EVALUATION")
    print("=================================================\n")

    print(f"MAE  : {mae:.2f}")

    print(f"RMSE : {rmse:.2f}")

    print(f"R2   : {r2:.4f}")