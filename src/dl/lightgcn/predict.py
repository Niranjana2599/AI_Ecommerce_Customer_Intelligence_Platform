import torch
import joblib
import pandas as pd
import numpy as np

from src.dl.lightgcn.model import LightGCN

from fastapi import HTTPException

from src.utils.path_config import MODELS_DIR

# =========================================================
# LOAD ENCODERS
# =========================================================

user_encoder = joblib.load(
    MODELS_DIR / "lightgcn_user_encoder.pkl"
)

product_encoder = joblib.load(
    MODELS_DIR / "lightgcn_product_encoder.pkl"
)

# =========================================================
# CONFIG
# =========================================================

NUM_USERS = len(
    user_encoder.classes_
)

NUM_ITEMS = len(
    product_encoder.classes_
)

# =========================================================
# LOAD MODEL
# =========================================================

model = LightGCN(

    num_users=NUM_USERS,

    num_items=NUM_ITEMS

)

model.load_state_dict(

    torch.load(
        "artifacts/models/lightgcn_model.pth",
        map_location=torch.device('cpu')
    )
)

model.eval()


# =========================================================
# RECOMMEND PRODUCTS
# =========================================================

def recommend_products(
    customer_id,
    top_k=10
):

    # Validate customer_id before encoding
    if customer_id not in user_encoder.classes_:

        return {

            "error": f"Unknown customer_id: {customer_id}"

        }
    
    if customer_id not in user_encoder.classes_:

         raise HTTPException(status_code=400,
                             detail=f"Unknown customer_id: {customer_id}" )

    encoded_user = user_encoder.transform(
        [customer_id]
    )[0]

    with torch.no_grad():

        user_embeddings = (
            model.user_embedding.weight
        )

        item_embeddings = (
            model.item_embedding.weight
        )

        user_vector = user_embeddings[
            encoded_user
        ]

        scores = torch.matmul(

            item_embeddings,

            user_vector

        )

        top_items = torch.topk(
            scores,
            k=top_k
        ).indices.numpy()

    recommended_products = (

        product_encoder.inverse_transform(
            top_items
        )
    )

    return {

        "customer_id": customer_id,

        "recommended_products":
            recommended_products.tolist()

    }