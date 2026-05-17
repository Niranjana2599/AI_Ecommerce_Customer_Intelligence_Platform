# pyright: reportMissingImports=false

from transformers import pipeline

from src.utils.path_config import MODELS_DIR

# =========================================================
# LOCAL MODEL PATH
# =========================================================

CLASSIFIER_model_path = (
    MODELS_DIR / "bart_mnli_classifier"
)

# =========================================================
# ZERO SHOT CLASSIFIER
# =========================================================

classifier = pipeline(

    "zero-shot-classification",

    model=str(CLASSIFIER_model_path)

)

# =========================================================
# REVIEW CLASSIFICATION
# =========================================================

def classify_review(text):

    labels = [

        "Delivery Issue",

        "Product Quality",

        "Payment Issue",

        "Customer Service",

        "Packaging",

        "Seller Experience"

    ]

    result = classifier(

        text,

        candidate_labels=labels

    )

    return {

        "review": text,

        "predicted_category":
            result['labels'][0],

        "confidence":
            round(result['scores'][0], 4)

    }

# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    sample = (
        "Delivery was delayed for 10 days"
    )

    result = classify_review(sample)

    print(result)