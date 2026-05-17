from src.dl.transformer.predict import (
    run_all_predictions
)


# =========================================================
# SIMPLE EVALUATION
# =========================================================

def evaluate_transformer():

    reviews = [

        "Excellent product quality",

        "Worst delivery experience",

        "Packaging was damaged",

        "Customer support was helpful"

    ]

    print("\n=================================================")
    print("TRANSFORMER EVALUATION")
    print("=================================================\n")

    for review in reviews:

        result = run_all_predictions(review)

        print(result)

        print("\n--------------------------------------\n")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    evaluate_transformer()