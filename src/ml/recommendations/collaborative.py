import joblib


# =========================================================
# LOAD PRODUCT SIMILARITY
# =========================================================

product_similarity_df = joblib.load(
    "artifacts/models/product_similarity.pkl"
)


# =========================================================
# COLLABORATIVE FILTERING
# =========================================================

def recommend_similar_products(product_id, top_n=5):

    try:

        similar_products = product_similarity_df[
            product_id
        ].sort_values(ascending=False)

        similar_products = similar_products.iloc[
            1:top_n+1
        ]

        recommendations = similar_products.index.tolist()

        return recommendations

    except Exception:

        return []