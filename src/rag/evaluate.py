from src.rag.predict import (

    ask_rag

)


# =========================================================
# EVALUATE RAG
# =========================================================

def evaluate_rag():

    questions = [

        "Which products have poor reviews?",

        "What causes delivery delay?",

        "Which customers are high value?"

    ]

    for question in questions:

        result = ask_rag(question)

        print("\n")

        print("=" * 50)

        print("QUESTION:", question)

        print("=" * 50)

        print(result['answer'])


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    evaluate_rag()