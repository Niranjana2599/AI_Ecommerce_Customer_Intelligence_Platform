import pandas as pd

from xgboost import XGBRegressor


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def create_features(df):

    df = df.copy()

    df['day'] = df.index.day

    df['month'] = df.index.month

    df['year'] = df.index.year

    df['weekday'] = df.index.weekday

    return df


# =========================================================
# TRAIN XGBOOST FORECAST MODEL
# =========================================================

def train_xgboost(df):

    df = create_features(df)

    X = df[[

        'day',
        'month',
        'year',
        'weekday'

    ]]

    y = df['Order_Count']

    model = XGBRegressor(

        n_estimators=100,

        learning_rate=0.05,

        max_depth=5,

        random_state=42

    )

    model.fit(X, y)

    return model