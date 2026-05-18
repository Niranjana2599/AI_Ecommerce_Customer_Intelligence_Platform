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

    print("\n========== ASK_RAG EXECUTED ==========\n")

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

    print("\n========== DOCUMENTS RETRIEVED ==========\n")

    print(f"Total Documents Retrieved: {len(docs)}")

    # =====================================================
    # BUILD CLEAN CONTEXT
    # =====================================================

    clean_context = ""

    for i, doc in enumerate(docs):

        clean_context += f"""

Retrieved Ecommerce Customer Data {i+1}:

{doc.page_content}

"""

    print("\n========== RETRIEVED CONTEXT ==========\n")

    print(clean_context)

    # =====================================================
    # GENERATE FINAL RESPONSE
    # =====================================================

    answer = generate_response(

        clean_context,

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
# TESTING
# =========================================================

if __name__ == "__main__":

    result = ask_rag(

        "Which customers are likely to churn?"

    )

    print("\n========== FINAL OUTPUT ==========\n")

    print(result)