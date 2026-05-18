import os

from src.rag.document_loader import (

    load_pdf,
    load_txt

)

from src.rag.text_chunking import (

    chunk_documents

)

from src.rag.vector_store import (

    create_vector_store,
    save_vector_store

)

# =========================================================
# BUILD RAG PIPELINE
# =========================================================

def build_rag_pipeline(file_path):

    extension = os.path.splitext(file_path)[1]

    # =====================================================
    # LOAD DOCUMENTS
    # =====================================================

    if extension == ".pdf":

        documents = load_pdf(file_path)

    elif extension == ".txt":

        documents = load_txt(file_path)

    else:

        raise ValueError(

            f"Unsupported file type: {extension}"

        )

    # =====================================================
    # CHECK EMPTY DOCUMENTS
    # =====================================================

    if not documents:

        raise ValueError(

            "No documents loaded."

        )

    # =====================================================
    # CHUNK DOCUMENTS
    # =====================================================

    chunks = chunk_documents(

        documents

    )

    # =====================================================
    # CREATE VECTOR STORE
    # =====================================================

    vector_db = create_vector_store(

        chunks

    )

    # =====================================================
    # SAVE VECTOR STORE
    # =====================================================

    save_vector_store(vector_db)

    return vector_db