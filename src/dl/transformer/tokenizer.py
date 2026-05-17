# pyright: reportMissingImports=false

from transformers import BertTokenizer


# =========================================================
# LOAD TOKENIZER
# =========================================================

def load_tokenizer():

    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased'
    )

    return tokenizer