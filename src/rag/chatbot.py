# =========================================================
# SMART RAG RESPONSE
# =========================================================

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