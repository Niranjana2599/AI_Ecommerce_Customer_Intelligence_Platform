import sys
import os

# =========================================================
# PROJECT ROOT PATH
# =========================================================

sys.path.append(

    os.path.abspath(

        os.path.join(

            os.path.dirname(__file__),

            "../.."

        )

    )

)

# =========================================================
# STREAMLIT
# =========================================================

import streamlit as st

# =========================================================
# RAG IMPORT
# =========================================================

from src.rag.predict import ask_rag

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(

    page_title="AI Ecommerce RAG Chatbot",

    layout="wide"

)

# =========================================================
# TITLE
# =========================================================

st.title(

    "AI Ecommerce Customer Intelligence RAG Chatbot"

)

st.markdown(

    """
Ask ecommerce business questions using your AI-powered RAG system.

Examples:
- Which customers are likely to churn?
- Which products have poor reviews?
- Summarize customer behavior patterns
- What are the high-risk customer segments?
"""
)

# =========================================================
# USER QUESTION
# =========================================================

question = st.text_input(

    "Enter your ecommerce question"

)

# =========================================================
# ASK AI BUTTON
# =========================================================

if st.button("Ask AI"):

    # =====================================================
    # VALIDATION
    # =====================================================

    if question.strip() == "":

        st.warning(

            "Please enter a question."

        )

    else:

        # =================================================
        # LOADING
        # =================================================

        with st.spinner(

            "Generating AI response..."

        ):

            try:

                print("\n========== STREAMLIT REQUEST ==========\n")

                print(question)

                # =========================================
                # GET RESPONSE
                # =========================================

                result = ask_rag(question)

                # =========================================
                # SUCCESS
                # =========================================

                st.success(

                    "AI Response Generated Successfully"

                )

                # =========================================
                # DISPLAY QUESTION
                # =========================================

                st.subheader("Question")

                st.write(

                    result["question"]

                )

                # =========================================
                # DISPLAY ANSWER
                # =========================================

                st.subheader("AI Answer")

                st.write(

                    result["answer"]

                )

            except Exception as e:
                
                import traceback
                
                st.error(f"Error occurred: {str(e)}")    
                
                st.text(traceback.format_exc())