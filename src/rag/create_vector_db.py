from src.rag.langchain_pipeline import (

    build_rag_pipeline

)


# =========================================================
# BUILD VECTOR DATABASE
# =========================================================

if __name__ == "__main__":

    file_path = "data/ecommerce_knowledge.txt"

    build_rag_pipeline(

        file_path

    )

    print(

        "Vector Database Created Successfully"

    )