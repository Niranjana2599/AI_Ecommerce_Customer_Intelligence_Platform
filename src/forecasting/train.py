import os
import joblib
import warnings

import pandas as pd
import numpy as np

from prophet import Prophet

from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

warnings.filterwarnings("ignore")


# =========================================================
# TRAIN FORECASTING MODELS
# =========================================================

def train_forecasting_models():

    print("\n=================================================")
    print("TIME SERIES FORECAST TRAINING STARTED")
    print("=================================================\n")

    # =====================================================
    # LOAD DATA
    # =====================================================

    DATA_PATH = "data/raw/orders.csv"

    orders = pd.read_csv(DATA_PATH)

    print(f"Orders Dataset Loaded: {orders.shape}")

    # =====================================================
    # CONVERT DATE
    # =====================================================

    orders['order_purchase_timestamp'] = pd.to_datetime(
        orders['order_purchase_timestamp']
    )

    # =====================================================
    # DAILY ORDER COUNT
    # =====================================================

    ts_df = orders[[
        'order_id',
        'order_purchase_timestamp'
    ]]

    daily_orders = (
        ts_df
        .groupby(
            ts_df['order_purchase_timestamp'].dt.date
        )
        .count()['order_id']
        .reset_index()
    )

    daily_orders.columns = [
        'Date',
        'Order_Count'
    ]

    daily_orders['Date'] = pd.to_datetime(
        daily_orders['Date']
    )

    daily_orders = daily_orders.sort_values(
        'Date'
    )

    daily_orders.set_index(
        'Date',
        inplace=True
    )

    print("\nDaily Orders Shape:")

    print(daily_orders.shape)

    # =====================================================
    # TRAIN TEST SPLIT
    # =====================================================

    train_size = int(len(daily_orders) * 0.8)

    train = daily_orders.iloc[:train_size]

    test = daily_orders.iloc[train_size:]

    # =====================================================
    # ARIMA MODEL
    # =====================================================

    print("\nTraining ARIMA Model...")

    arima_model = SARIMAX(

        train['Order_Count'],

        order=(5,1,2),

        seasonal_order=(1,1,1,7)

    )

    arima_result = arima_model.fit()

    # =====================================================
    # ARIMA FORECAST
    # =====================================================

    arima_preds = arima_result.predict(

        start=len(train),

        end=len(train)+len(test)-1,

        dynamic=False

    )

    arima_mae = mean_absolute_error(
        test['Order_Count'],
        arima_preds
    )

    arima_rmse = np.sqrt(
        mean_squared_error(
            test['Order_Count'],
            arima_preds
        )
    )

    print("\nARIMA PERFORMANCE")

    print(f"MAE  : {arima_mae:.2f}")

    print(f"RMSE : {arima_rmse:.2f}")

    # =====================================================
    # PREPARE FOR PROPHET
    # =====================================================

    prophet_df = (
        daily_orders
        .reset_index()[['Date', 'Order_Count']]
    )

    prophet_df.columns = ['ds', 'y']

    # =====================================================
    # PROPHET TRAIN TEST SPLIT
    # =====================================================

    prophet_train_size = int(
        len(prophet_df) * 0.8
    )

    prophet_train = prophet_df.iloc[
        :prophet_train_size
    ]

    prophet_test = prophet_df.iloc[
        prophet_train_size:
    ]

    # =====================================================
    # TRAIN PROPHET
    # =====================================================

    print("\nTraining Prophet Model...")

    prophet_model = Prophet(

        daily_seasonality=True,

        weekly_seasonality=True,

        yearly_seasonality=True

    )

    prophet_model.fit(prophet_train)

    # =====================================================
    # FORECAST
    # =====================================================

    future = prophet_model.make_future_dataframe(

        periods=len(prophet_test)

    )

    forecast = prophet_model.predict(future)

    prophet_preds = forecast.iloc[
        -len(prophet_test):
    ]

    # =====================================================
    # EVALUATE PROPHET
    # =====================================================

    prophet_mae = mean_absolute_error(

        prophet_test['y'],

        prophet_preds['yhat']

    )

    prophet_rmse = np.sqrt(

        mean_squared_error(

            prophet_test['y'],

            prophet_preds['yhat']

        )
    )

    print("\nPROPHET PERFORMANCE")

    print(f"MAE  : {prophet_mae:.2f}")

    print(f"RMSE : {prophet_rmse:.2f}")

    # =====================================================
    # FUTURE FORECAST
    # =====================================================

    future_30 = prophet_model.make_future_dataframe(
        periods=30
    )

    future_forecast = prophet_model.predict(
        future_30
    )

    # =====================================================
    # CREATE ARTIFACTS DIRECTORY
    # =====================================================

    os.makedirs(
        "artifacts/models",
        exist_ok=True
    )

    # =====================================================
    # SAVE MODELS
    # =====================================================

    joblib.dump(

        arima_result,

        "artifacts/models/arima_forecast_model.pkl"

    )

    joblib.dump(

        prophet_model,

        "artifacts/models/prophet_forecast_model.pkl"

    )

    # =====================================================
    # SAVE FORECAST CSV
    # =====================================================

    future_forecast.to_csv(

        "artifacts/forecast_results.csv",

        index=False

    )

    print("\n=================================================")
    print("FORECAST MODELS SAVED SUCCESSFULLY")
    print("=================================================\n")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    train_forecasting_models()