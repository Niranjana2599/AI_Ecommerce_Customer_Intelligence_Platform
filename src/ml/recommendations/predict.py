import joblib

from src.ml.recommendations.hybrid import (
    hybrid_recommendation
)


# =========================================================
# LOAD ARTIFACTS
# =========================================================

from src.utils.path_config import MODELS_DIR

product_similarity_df = joblib.load(
    MODELS_DIR / "product_similarity.pkl"
)


# =========================================================
# MAIN PREDICT FUNCTION
# =========================================================

def recommend_products(product_id, top_n=5):

    recommendations = hybrid_recommendation(
        product_id,
        top_n
    )

    return {

        "product_id": str(product_id),

        "recommended_products": [

            str(x) for x in recommendations

        ]
    }


# =========================================================
# TESTING
# =========================================================

if __name__ == "__main__":

    sample_product = product_similarity_df.index[0]

    result = recommend_products(sample_product)

    print("\nRecommendation Result:\n")

    print(result)