import os
import pickle
import pandas as pd

from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# FUNCTION: TRAIN CLV MODEL
# =========================================================

def run_training():

    print("\n=================================================")
    print("CLV MODEL TRAINING STARTED")
    print("=================================================\n")

    # =====================================================
    # LOAD DATA
    # =====================================================

    DATA_PATH = "data/processed/statistical_analysis_dataset.csv"

    print("Loading dataset...\n")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset Loaded Successfully: {df.shape}")

    # =====================================================
    # TARGET COLUMN
    # =====================================================

    TARGET_COLUMN = "Monetary"

    # =====================================================
    # CHECK TARGET
    # =====================================================

    if TARGET_COLUMN not in df.columns:

        raise Exception(
            f"\nTarget column '{TARGET_COLUMN}' not found.\n"
        )

    # =====================================================
    # DROP COLUMNS
    # =====================================================

    DROP_COLUMNS = [

        # IDs
        "order_id",
        "customer_id",
        "customer_unique_id",
        "product_id",
        "seller_id",
        "review_id",

        # Dates
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "shipping_limit_date",
        "review_creation_date",
        "review_answer_timestamp",

        # Leakage
        "Monetary"

    ]

    # =====================================================
    # FEATURE / TARGET SPLIT
    # =====================================================

    X = df.drop(columns=DROP_COLUMNS)

    y = df[TARGET_COLUMN]

    print("\nFeature Shape:", X.shape)

    print("Target Shape:", y.shape)

    # =====================================================
    # HANDLE CATEGORICAL COLUMNS
    # =====================================================

    categorical_cols = X.select_dtypes(include=['object']).columns

    print("\nCategorical Columns:\n")

    print(categorical_cols.tolist())

    encoders = {}

    for col in categorical_cols:

        le = LabelEncoder()

        X[col] = le.fit_transform(X[col].astype(str))

        encoders[col] = le

    # =====================================================
    # CREATE ARTIFACT DIRECTORY
    # =====================================================

    os.makedirs("artifacts/models", exist_ok=True)

    # =====================================================
    # SAVE ENCODERS
    # =====================================================

    with open("artifacts/models/clv_encoders.pkl", "wb") as f:

        pickle.dump(encoders, f)

    # =====================================================
    # SAVE FEATURE NAMES
    # =====================================================

    feature_columns = X.columns.tolist()

    with open("artifacts/models/clv_features.pkl", "wb") as f:

        pickle.dump(feature_columns, f)

    print("\nFeature Names Saved Successfully.")

    # =====================================================
    # TRAIN TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nTrain Shape:", X_train.shape)

    print("Test Shape:", X_test.shape)

    # =====================================================
    # MODEL TRAINING
    # =====================================================

    print("\nTraining LightGBM Regressor...\n")

    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("\nModel Training Completed Successfully.")

    # =====================================================
    # PREDICTIONS
    # =====================================================

    y_pred = model.predict(X_test)

    # =====================================================
    # EVALUATION
    # =====================================================

    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = mse ** 0.5

    r2 = r2_score(y_test, y_pred)

    print("\n=================================================")
    print("MODEL PERFORMANCE")
    print("=================================================\n")

    print(f"MAE  : {mae:.4f}")

    print(f"MSE  : {mse:.4f}")

    print(f"RMSE : {rmse:.4f}")

    print(f"R2   : {r2:.4f}")

    # =====================================================
    # SAVE MODEL
    # =====================================================

    model_path = "artifacts/models/clv_model.pkl"

    with open(model_path, "wb") as f:

        pickle.dump(model, f)

    print("\n=================================================")
    print("MODEL SAVED SUCCESSFULLY")
    print("=================================================\n")

    print(f"Saved Path: {model_path}")

    print("\n=================================================")
    print("TRAINING COMPLETED")
    print("=================================================\n")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    run_training()