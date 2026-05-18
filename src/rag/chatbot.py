from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).resolve().parents[2] / ".env"

load_dotenv(dotenv_path=env_path)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "AI_Ecommerce_RAG"

print("LANGCHAIN_API_KEY:", os.getenv("LANGCHAIN_API_KEY"))

# =========================================================
# SMART RAG RESPONSE
# =========================================================

from langsmith import traceable


@traceable(name="RAG_Response_Generation")
def generate_response(

    context,

    question

):

    question = question.lower()

    sentences = context.split("\n")

    relevant_sentences = []


    # =====================================================
    # SIMPLE KEYWORD MATCHING
    # =====================================================

    for sentence in sentences:

        sentence_lower = sentence.lower()

        if any(

            word in sentence_lower

            for word in question.split()

        ):

            relevant_sentences.append(

                sentence.strip()

            )


    # =====================================================
    # FALLBACK
    # =====================================================

    if not relevant_sentences:

        relevant_sentences.append(

            "No relevant information found."

        )


    final_answer = "\n".join(

        relevant_sentences

    )

    return final_answer