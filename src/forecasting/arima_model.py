from statsmodels.tsa.statespace.sarimax import SARIMAX


# =========================================================
# TRAIN ARIMA MODEL
# =========================================================

def train_arima(train_series):

    model = SARIMAX(

        train_series,

        order=(5,1,2),

        seasonal_order=(1,1,1,7)

    )

    result = model.fit()

    return result