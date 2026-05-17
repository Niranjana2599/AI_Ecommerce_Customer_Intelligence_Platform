# pyright: reportMissingImports=false

from transformers import pipeline

from src.utils.path_config import MODELS_DIR

# =========================================================
# LOCAL MODEL PATH
# =========================================================

SUMMARIZER_model_path = (
    MODELS_DIR / "gpt2_summarizer"
)

# =========================================================
# LOAD GENERATOR
# =========================================================

generator = pipeline(

    "text-generation",

    model=str(SUMMARIZER_model_path)

)

# =========================================================
# SUMMARIZE REVIEW
# =========================================================

def summarize_review(text):

    prompt = f"Summarize this review briefly:\n{text}"

    result = generator(

        prompt,

        max_new_tokens=40,

        truncation=True

    )

    return {

        "original_text": text,

        "summary": result[0]['generated_text']

    }

# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    sample = (
        "Delivery was delayed but the product quality was excellent."
    )

    print(
        summarize_review(sample)
    )