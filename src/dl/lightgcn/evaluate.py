import torch
import joblib
import pandas as pd
import numpy as np

from src.dl.lightgcn.model import LightGCN


# =========================================================
# LOAD ENCODERS
# =========================================================

user_encoder = joblib.load(
    "artifacts/models/lightgcn_user_encoder.pkl"
)

product_encoder = joblib.load(
    "artifacts/models/lightgcn_product_encoder.pkl"
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
# RECALL@K
# =========================================================

def recall_at_k(

    recommended_items,

    relevant_items,

    k=10

):

    recommended_items = recommended_items[:k]

    hits = len(

        set(recommended_items)
        &
        set(relevant_items)

    )

    recall = hits / len(relevant_items)

    return recall


# =========================================================
# EVALUATION
# =========================================================

def evaluate_model():

    print("\n=================================================")
    print("LIGHTGCN EVALUATION")
    print("=================================================\n")

    DATA_PATH = (
        "data/processed/statistical_analysis_dataset.csv"
    )

    df = pd.read_csv(DATA_PATH)

    interaction_df = df[[

        'customer_unique_id',

        'product_id'

    ]].dropna()

    sample_users = interaction_df[
        'customer_unique_id'
    ].unique()[:100]

    recalls = []

    for customer_id in sample_users:

        try:

            encoded_user = user_encoder.transform(
                [customer_id]
            )[0]

            user_vector = (
                model.user_embedding.weight[
                    encoded_user
                ]
            )

            item_embeddings = (
                model.item_embedding.weight
            )

            scores = torch.matmul(

                item_embeddings,

                user_vector

            )

            top_items = torch.topk(

                scores,

                k=10

            ).indices.numpy()

            recommended_products = (

                product_encoder.inverse_transform(
                    top_items
                )
            )

            actual_products = interaction_df[
                interaction_df[
                    'customer_unique_id'
                ] == customer_id
            ]['product_id'].unique()

            recall = recall_at_k(

                recommended_products,

                actual_products,

                k=10

            )

            recalls.append(recall)

        except:

            continue

    avg_recall = np.mean(recalls)

    print(f"Recall@10 : {avg_recall:.4f}")

    print("\n=================================================")
    print("EVALUATION COMPLETED")
    print("=================================================\n")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    evaluate_model()