import os
import joblib
import random

import torch
import torch.optim as optim

import pandas as pd
import numpy as np

from scipy.sparse import coo_matrix

from sklearn.preprocessing import LabelEncoder

from src.dl.lightgcn.model import LightGCN

from src.dl.lightgcn.utils import (
    bpr_loss,
    negative_sampling
)


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)


# =========================================================
# TRAIN LIGHTGCN
# =========================================================

def train_lightgcn():

    print("\n=================================================")
    print("LIGHTGCN TRAINING STARTED")
    print("=================================================\n")

    DATA_PATH = (
        "data/processed/statistical_analysis_dataset.csv"
    )

    master_df = pd.read_csv(DATA_PATH)

    interaction_df = master_df[[

        'customer_unique_id',

        'product_id',

        'review_score'

    ]].dropna()

    # =====================================================
    # LABEL ENCODING
    # =====================================================

    user_encoder = LabelEncoder()

    product_encoder = LabelEncoder()

    interaction_df['user_id'] = (

        user_encoder.fit_transform(
            interaction_df['customer_unique_id']
        )
    )

    interaction_df['item_id'] = (

        product_encoder.fit_transform(
            interaction_df['product_id']
        )
    )

    # =====================================================
    # SAVE ENCODERS
    # =====================================================

    os.makedirs(
        "artifacts/models",
        exist_ok=True
    )

    joblib.dump(

        user_encoder,

        "artifacts/models/lightgcn_user_encoder.pkl"

    )

    joblib.dump(

        product_encoder,

        "artifacts/models/lightgcn_product_encoder.pkl"

    )

    # =====================================================
    # GRAPH CREATION
    # =====================================================

    num_users = interaction_df[
        'user_id'
    ].nunique()

    num_items = interaction_df[
        'item_id'
    ].nunique()

    user_nodes = interaction_df[
        'user_id'
    ].values

    item_nodes = (

        interaction_df['item_id'].values
        + num_users

    )

    edge_index = torch.tensor(

        [
            np.concatenate([
                user_nodes,
                item_nodes
            ]),

            np.concatenate([
                item_nodes,
                user_nodes
            ])
        ],

        dtype=torch.long

    ).to(device)

    # =====================================================
    # MODEL
    # =====================================================

    model = LightGCN(

        num_users=num_users,

        num_items=num_items,

        embedding_dim=64,

        num_layers=3

    ).to(device)

    optimizer = optim.Adam(

        model.parameters(),

        lr=0.001

    )

    # =====================================================
    # NEGATIVE SAMPLING
    # =====================================================

    all_items = set(
        interaction_df['item_id'].unique()
    )

    user_item_set = set(

        zip(

            interaction_df['user_id'],

            interaction_df['item_id']

        )
    )

    # =====================================================
    # TRAINING
    # =====================================================

    epochs = 20

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        user_embeds, item_embeds = model(
            edge_index
        )

        users, pos_items, neg_items = (

            negative_sampling(

                interaction_df,

                all_items,

                user_item_set

            )
        )

        users = users.to(device)

        pos_items = pos_items.to(device)

        neg_items = neg_items.to(device)

        loss = bpr_loss(

            user_embeds,

            item_embeds,

            users,

            pos_items,

            neg_items

        )

        loss.backward()

        optimizer.step()

        print(
            f"Epoch {epoch+1}/{epochs}"
            f" | Loss: {loss.item():.4f}"
        )

    # =====================================================
    # SAVE MODEL
    # =====================================================

    torch.save(

        model.state_dict(),

        "artifacts/models/lightgcn_model.pth"

    )

    print("\n=================================================")
    print("LIGHTGCN MODEL SAVED")
    print("=================================================\n")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    train_lightgcn()