import os
import joblib
import pandas as pd

from scipy.sparse import csr_matrix

from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# TRAIN RECOMMENDATION SYSTEM
# =========================================================


def train_recommendation_system():

    print("\n=================================================")
    print("RECOMMENDATION MODEL TRAINING STARTED")
    print("=================================================\n")

    # =====================================================
    # LOAD DATA
    # =====================================================

    DATA_PATH = "data/processed/statistical_analysis_dataset.csv"

    master_df = pd.read_csv(DATA_PATH)

    print(f"Dataset Loaded Successfully: {master_df.shape}")

    # =====================================================
    # CUSTOMER PRODUCT INTERACTION
    # =====================================================

    interaction_df = master_df[[
        'customer_unique_id',
        'product_id',
        'review_score'
    ]]

    interaction_df = interaction_df.dropna()

    print("\nInteraction Shape:")
    print(interaction_df.shape)

    # =====================================================
    # USER ITEM MATRIX
    # =====================================================

    user_item_matrix = interaction_df.pivot_table(
        index='customer_unique_id',
        columns='product_id',
        values='review_score',
        fill_value=0
    )

    print("\nUser Item Matrix Shape:")
    print(user_item_matrix.shape)

    # =====================================================
    # SPARSE MATRIX
    # =====================================================

    sparse_matrix = csr_matrix(user_item_matrix)

    print("\nSparse Matrix Created")

    # =====================================================
    # PRODUCT SIMILARITY
    # =====================================================

    product_similarity = cosine_similarity(
        user_item_matrix.T
    )

    product_similarity_df = pd.DataFrame(
        product_similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )

    print("\nProduct Similarity Matrix Created")

    # =====================================================
    # CONTENT DATAFRAME
    # =====================================================

    content_df = master_df[[
        'product_id',
        'product_category_name'
    ]].drop_duplicates()

    content_df = content_df.dropna()

    print("\nContent Data Shape:")
    print(content_df.shape)

    # =====================================================
    # CREATE ARTIFACT FOLDER
    # =====================================================

    os.makedirs("artifacts/models", exist_ok=True)

    # =====================================================
    # SAVE ARTIFACTS
    # =====================================================

    joblib.dump(
        user_item_matrix,
        "artifacts/models/user_item_matrix.pkl"
    )

    joblib.dump(
        product_similarity_df,
        "artifacts/models/product_similarity.pkl"
    )

    joblib.dump(
        content_df,
        "artifacts/models/content_df.pkl"
    )

    print("\n=================================================")
    print("RECOMMENDATION ARTIFACTS SAVED")
    print("=================================================\n")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    train_recommendation_system()