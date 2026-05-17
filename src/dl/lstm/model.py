from tensorflow.keras.models import Sequential # type: ignore

from tensorflow.keras.layers import ( # type: ignore
    Dense,
    LSTM,
    Dropout
) 


# =========================================================
# BUILD LSTM MODEL
# =========================================================

def build_lstm_model(input_shape):

    model = Sequential()

    model.add(

        LSTM(

            128,

            return_sequences=True,

            input_shape=input_shape

        )
    )

    model.add(Dropout(0.2))

    model.add(

        LSTM(64)

    )

    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(

        optimizer='adam',

        loss='mse'

    )

    return model