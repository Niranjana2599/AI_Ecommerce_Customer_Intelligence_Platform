import os
import joblib

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.callbacks import (
    EarlyStopping
)

from src.dl.lstm.model import (
    build_lstm_model
)

from src.dl.lstm.utils import (
    create_sequences
)


# =========================================================
# TRAIN LSTM FORECAST MODEL
# =========================================================

def train_lstm_forecasting():

    print("\n=================================================")
    print("LSTM FORECAST TRAINING STARTED")
    print("=================================================\n")

    DATA_PATH = "data/raw/orders.csv"

    orders = pd.read_csv(DATA_PATH)

    # =====================================================
    # DATE CONVERSION
    # =====================================================

    orders['order_purchase_timestamp'] = pd.to_datetime(

        orders['order_purchase_timestamp']

    )

    # =====================================================
    # DAILY ORDERS
    # =====================================================

    daily_orders = (

        orders.groupby(

            orders['order_purchase_timestamp'].dt.date

        )

        .size()

        .reset_index(name='Order_Count')

    )

    daily_orders.columns = [

        'Date',
        'Order_Count'

    ]

    # =====================================================
    # SCALING
    # =====================================================

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(

        daily_orders[['Order_Count']]

    )

    # =====================================================
    # SAVE SCALER
    # =====================================================

    os.makedirs(

        "artifacts/models",

        exist_ok=True

    )

    joblib.dump(

        scaler,

        "artifacts/models/lstm_scaler.pkl"

    )

    # =====================================================
    # CREATE SEQUENCES
    # =====================================================

    X, y = create_sequences(

        scaled_data,

        seq_length=10

    )

    X = X.reshape(

        X.shape[0],

        X.shape[1],

        1

    )

    # =====================================================
    # MODEL
    # =====================================================

    model = build_lstm_model(

        (X.shape[1], 1)

    )

    early_stop = EarlyStopping(

        patience=5,

        restore_best_weights=True

    )

    model.fit(

        X,
        y,

        epochs=20,

        batch_size=32,

        validation_split=0.2,

        callbacks=[early_stop],

        verbose=1

    )

    # =====================================================
    # SAVE MODEL
    # =====================================================

    model.save(

        "artifacts/models/lstm_forecasting_model.h5"

    )

    print("\n=================================================")
    print("LSTM FORECAST MODEL SAVED")
    print("=================================================\n")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    train_lstm_forecasting()