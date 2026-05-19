from dotenv import load_dotenv
from pathlib import Path
import os

# =========================================================
# LOAD ENV VARIABLES
# =========================================================

env_path = Path(__file__).resolve().parents[2] / ".env"

load_dotenv(dotenv_path=env_path)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "AI_Ecommerce_RAG"

print("LANGCHAIN_API_KEY:", os.getenv("LANGCHAIN_API_KEY"))

# =========================================================
# LANGSMITH
# =========================================================

from langsmith import traceable

# =========================================================
# OLLAMA LLM
# =========================================================

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="phi3",
    base_url=os.getenv("OLLAMA_BASE_URL"),
    temperature=0.3
)


# =========================================================
# PROMPT TEMPLATE
# =========================================================

from langchain_core.prompts import PromptTemplate

template = """
You are an AI Ecommerce Business Analyst.

Analyze the retrieved ecommerce customer data carefully.

Provide a concise, business-friendly, human-readable answer.

Avoid dumping raw numerical records directly.

If the question is related to churn:
- identify customer behavior patterns
- mention risky customer groups
- summarize insights clearly

If the answer is not available in the context,
say:
"I could not find enough relevant information."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(

    input_variables=[

        "context",

        "question"

    ],

    template=template

)

# =========================================================
# RESPONSE GENERATION
# =========================================================

@traceable(name="RAG_Response_Generation")
def generate_response(context, question):

    print("\n========== RAG GENERATION STARTED ==========\n")

    final_prompt = prompt.format(

        context=context,

        question=question

    )

    response = llm.invoke(final_prompt)

    print("\n========== GENERATED RESPONSE ==========\n")

    print(response.content)

    return response.content