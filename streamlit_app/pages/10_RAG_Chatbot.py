import sys
import os

sys.path.append(

    os.path.abspath(

        os.path.join(

            os.path.dirname(__file__),

            "../.."

        )

    )

)

import streamlit as st

from src.rag.predict import (

    ask_rag

)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(

    page_title="RAG Chatbot",

    layout="wide"

)


# =========================================================
# TITLE
# =========================================================

st.title(

    "AI Ecommerce RAG Chatbot"

)

st.markdown(

    "Ask questions about ecommerce business insights"

)


# =========================================================
# USER INPUT
# =========================================================

question = st.text_input(

    "Enter your question"

)


# =========================================================
# ASK QUESTION
# =========================================================

if st.button("Ask AI"):

    with st.spinner(

        "Generating response..."

    ):

        result = ask_rag(question)

        st.success(

            "Response Generated Successfully"
        )

        st.subheader("Question")

        st.write(

            result['question']

        )

        st.subheader("Answer")

        st.write(

            result['answer']

        )