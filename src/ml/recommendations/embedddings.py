import joblib
import numpy as np


# =========================================================
# LOAD MATRIX
# =========================================================

user_item_matrix = joblib.load(
    "artifacts/models/user_item_matrix.pkl"
)


# =========================================================
# SIMPLE PRODUCT EMBEDDINGS
# =========================================================

product_embeddings = np.array(
    user_item_matrix.T.values
)


# =========================================================
# GET PRODUCT EMBEDDING
# =========================================================


def get_product_embedding(product_id):

    if product_id not in user_item_matrix.columns:

        return None

    idx = list(user_item_matrix.columns).index(product_id)

    return product_embeddings[idx]