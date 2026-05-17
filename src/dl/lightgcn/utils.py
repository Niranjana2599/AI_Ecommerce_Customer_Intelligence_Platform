import random
import torch


# =========================================================
# BPR LOSS
# =========================================================

def bpr_loss(
    user_embeds,
    item_embeds,
    users,
    pos_items,
    neg_items
):

    user_vector = user_embeds[users]

    pos_vector = item_embeds[pos_items]

    neg_vector = item_embeds[neg_items]

    pos_scores = (
        user_vector * pos_vector
    ).sum(dim=1)

    neg_scores = (
        user_vector * neg_vector
    ).sum(dim=1)

    loss = -torch.mean(

        torch.log(
            torch.sigmoid(
                pos_scores - neg_scores
            )
        )
    )

    return loss


# =========================================================
# NEGATIVE SAMPLING
# =========================================================

def negative_sampling(

    interaction_df,

    all_items,

    user_item_set

):

    users = []
    pos_items = []
    neg_items = []

    for user, pos_item in user_item_set:

        neg_item = random.choice(
            list(all_items)
        )

        while (
            user,
            neg_item
        ) in user_item_set:

            neg_item = random.choice(
                list(all_items)
            )

        users.append(user)

        pos_items.append(pos_item)

        neg_items.append(neg_item)

    return (

        torch.LongTensor(users),

        torch.LongTensor(pos_items),

        torch.LongTensor(neg_items)

    )