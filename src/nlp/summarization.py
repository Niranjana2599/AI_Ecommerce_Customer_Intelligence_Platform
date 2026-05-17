# =========================================================
# TEMPORARY SIMPLE SUMMARIZER
# =========================================================

def summarize_review(text):

    if text is None:
        return ""

    text = str(text)

    return text[:200]