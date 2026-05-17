# pyright: reportMissingImports=false

from sentence_transformers import (
    SentenceTransformer
)

import numpy as np


# =========================================================
# LOAD MODEL
# =========================================================

embedding_model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)


# =========================================================
# GENERATE EMBEDDINGS
# =========================================================

def generate_embeddings(text):

    embedding = embedding_model.encode(text)

    return {

        "text": text,

        "embedding_dimension": len(embedding),

        "embedding_vector": embedding.tolist()

    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    sample = "Excellent product quality"

    result = generate_embeddings(sample)

    print(result)