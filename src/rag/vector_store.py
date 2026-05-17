import os

from langchain_community.vectorstores import FAISS

from src.rag.embedding_model import (
    load_embedding_model
)

from src.utils.path_config import VECTOR_DB_DIR

# =========================================================
# CREATE VECTOR STORE
# =========================================================

def create_vector_store(chunks):

    embeddings = load_embedding_model()

    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_db


# =========================================================
# SAVE VECTOR STORE
# =========================================================

def save_vector_store(
    vector_db,
    save_path=VECTOR_DB_DIR
):

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    vector_db.save_local(save_path)


# =========================================================
# LOAD VECTOR STORE
# =========================================================

def load_vector_store(
    save_path=VECTOR_DB_DIR
):

    embeddings = load_embedding_model()

    vector_db = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_db