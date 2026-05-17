from src.rag.vector_store import (

    load_vector_store

)

from src.rag.retriever import (

    create_retriever

)

from src.rag.chatbot import (

    generate_response

)

# =========================================================
# RAG QUESTION ANSWERING
# =========================================================

def ask_rag(question):

    # =====================================================
    # LOAD VECTOR DATABASE
    # =====================================================

    vector_db = load_vector_store()

    # =====================================================
    # CREATE RETRIEVER
    # =====================================================

    retriever = create_retriever(
        vector_db
    )

    # =====================================================
    # RETRIEVE DOCUMENTS
    # =====================================================

    docs = retriever.invoke(question)

    # =====================================================
    # BUILD CONTEXT
    # =====================================================

    context = "\n".join(

        [doc.page_content for doc in docs]

    )

    # =====================================================
    # GENERATE RESPONSE
    # =====================================================

    answer = generate_response(

        context,

        question

    )

    # =====================================================
    # RETURN RESULT
    # =====================================================

    return {

        "question": question,

        "answer": answer

    }

# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    result = ask_rag(

        "What products have poor reviews?"

    )

    print(result)