from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# EVALUATE MODEL
# =========================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = mse ** 0.5

    r2 = r2_score(y_test, predictions)

    print("\n=================================================")
    print("DELIVERY DELAY MODEL EVALUATION")
    print("=================================================\n")

    print(f"MAE  : {mae:.4f}")

    print(f"MSE  : {mse:.4f}")

    print(f"RMSE : {rmse:.4f}")

    print(f"R2   : {r2:.4f}")