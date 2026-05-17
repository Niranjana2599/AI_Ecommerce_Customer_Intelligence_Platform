# pyright: reportMissingImports=false

from transformers import (
    BertForSequenceClassification
)

import torch


# =========================================================
# LOAD BERT MODEL
# =========================================================

def load_bert_model():

    model = BertForSequenceClassification.from_pretrained(

        'bert-base-uncased',

        num_labels=3

    )

    return model