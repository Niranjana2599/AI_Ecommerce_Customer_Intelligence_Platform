import joblib


# =========================================================
# LOAD PROPHET MODEL
# =========================================================

prophet_model = joblib.load(
    "artifacts/models/prophet_forecast_model.pkl"
)


# =========================================================
# FORECAST FUNCTION
# =========================================================

def forecast_orders(days=30):

    future = prophet_model.make_future_dataframe(
        periods=days
    )

    forecast = prophet_model.predict(
        future
    )

    results = forecast[[

        'ds',
        'yhat',
        'yhat_lower',
        'yhat_upper'

    ]].tail(days)

    return results.to_dict(
        orient='records'
    )


# =========================================================
# TESTING
# =========================================================

if __name__ == "__main__":

    result = forecast_orders(7)

    print("\nForecast Result:\n")

    for row in result:

        print(row)