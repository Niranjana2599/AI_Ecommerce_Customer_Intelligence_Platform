import os
import pickle
import pandas as pd

from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# FUNCTION: TRAIN CHURN MODEL
# =========================================================

def run_training():

    print("\n=================================================")
    print("CHURN MODEL TRAINING STARTED")
    print("=================================================\n")

    # =====================================================
    # LOAD DATA
    # =====================================================

    DATA_PATH = "data/processed/statistical_analysis_dataset.csv"

    print("Loading dataset...\n")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset Loaded Successfully: {df.shape}")

    # =====================================================
    # DISPLAY COLUMNS
    # =====================================================

    print("\n================ DATASET COLUMNS ================\n")

    print(df.columns.tolist())

    print("\n=================================================\n")

    # =====================================================
    # TARGET COLUMN
    # =====================================================

    TARGET_COLUMN = "churn"

    # =====================================================
    # CHECK TARGET EXISTS
    # =====================================================

    if TARGET_COLUMN not in df.columns:

        raise Exception(
            f"\nTarget column '{TARGET_COLUMN}' not found.\n"
        )

    # =====================================================
    # DROP UNNECESSARY COLUMNS
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

    # Leakage Columns
    "Recency",
    "Frequency",
    "Monetary",

    "total_orders",
    "total_spent",
    "avg_order_value",

    "avg_review_score",
    "avg_delivery_delay",

    "unique_products_purchased",

    "product_revenue",
    "product_total_orders",
    "product_avg_review",
    "product_avg_delay",

    "seller_total_revenue",
    "seller_total_orders",
    "seller_avg_review",
    "seller_avg_delay"

]

    # =====================================================
    # FEATURE / TARGET SPLIT
    # =====================================================

    X = df.drop(columns=[TARGET_COLUMN] + DROP_COLUMNS)

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
    # CREATE ARTIFACT FOLDER
    # =====================================================

    os.makedirs("artifacts/models", exist_ok=True)

    # =====================================================
    # SAVE ENCODERS
    # =====================================================

    with open("artifacts/models/churn_encoders.pkl", "wb") as f:

        pickle.dump(encoders, f)

    print("\nEncoders Saved Successfully.")

    # =====================================================
    # SAVE FEATURE NAMES
    # =====================================================

    feature_columns = X.columns.tolist()

    with open("artifacts/models/churn_features.pkl", "wb") as f:

        pickle.dump(feature_columns, f)

    print("\nFeature Names Saved Successfully.")

    # =====================================================
    # TRAIN TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain Shape:", X_train.shape)

    print("Test Shape:", X_test.shape)

    # =====================================================
    # MODEL TRAINING
    # =====================================================

    print("\nTraining LightGBM Model...\n")

    model = LGBMClassifier(
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

    y_prob = model.predict_proba(X_test)[:, 1]

    # =====================================================
    # EVALUATION
    # =====================================================

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n=================================================")
    print("MODEL PERFORMANCE")
    print("=================================================\n")

    print(f"Accuracy  : {accuracy:.4f}")

    print(f"Precision : {precision:.4f}")

    print(f"Recall    : {recall:.4f}")

    print(f"F1 Score  : {f1:.4f}")

    print(f"ROC AUC   : {roc_auc:.4f}")

    print("\n=================================================")
    print("CLASSIFICATION REPORT")
    print("=================================================\n")

    print(classification_report(y_test, y_pred))

    print("\n=================================================")
    print("CONFUSION MATRIX")
    print("=================================================\n")

    print(confusion_matrix(y_test, y_pred))

    # =====================================================
    # SAVE MODEL
    # =====================================================

    model_path = "artifacts/models/churn_model.pkl"

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