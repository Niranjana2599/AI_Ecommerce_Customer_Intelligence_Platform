from prophet import Prophet


# =========================================================
# TRAIN PROPHET MODEL
# =========================================================

def train_prophet(prophet_df):

    model = Prophet(

        daily_seasonality=True,

        weekly_seasonality=True,

        yearly_seasonality=True

    )

    model.fit(prophet_df)

    return model