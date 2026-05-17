import torch
import torch.nn as nn

from torch_geometric.nn import LGConv


# =========================================================
# LIGHTGCN MODEL
# =========================================================

class LightGCN(nn.Module):

    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=64,
        num_layers=3
    ):

        super().__init__()

        self.num_users = num_users
        self.num_items = num_items

        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim
        )

        self.item_embedding = nn.Embedding(
            num_items,
            embedding_dim
        )

        self.convs = nn.ModuleList()

        for _ in range(num_layers):

            self.convs.append(LGConv())

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, edge_index):

        user_emb = self.user_embedding.weight

        item_emb = self.item_embedding.weight

        x = torch.cat([user_emb, item_emb])

        embeddings = [x]

        for conv in self.convs:

            x = conv(x, edge_index)

            embeddings.append(x)

        embeddings = torch.stack(
            embeddings,
            dim=0
        )

        final_embeddings = embeddings.mean(dim=0)

        final_user_embeddings = final_embeddings[
            :self.num_users
        ]

        final_item_embeddings = final_embeddings[
            self.num_users:
        ]

        return (
            final_user_embeddings,
            final_item_embeddings
        )