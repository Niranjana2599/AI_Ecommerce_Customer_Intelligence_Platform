from src.rag.document_loader import (

    load_pdf

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

    documents = load_pdf(file_path)

    chunks = chunk_documents(

        documents

    )

    vector_db = create_vector_store(

        chunks

    )

    save_vector_store(vector_db)

    return vector_db