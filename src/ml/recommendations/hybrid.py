from src.ml.recommendations.collaborative import (
    recommend_similar_products
)

from src.ml.recommendations.content_based import (
    recommend_content_based
)


# =========================================================
# HYBRID RECOMMENDATION
# =========================================================

def hybrid_recommendation(product_id, top_n=5):

    collaborative_results = recommend_similar_products(
        product_id,
        top_n
    )

    content_results = recommend_content_based(
        product_id,
        top_n
    )

    recommendations = list(
        dict.fromkeys(
            collaborative_results + content_results
        )
    )

    return recommendations[:top_n]