import joblib


# =========================================================
# LOAD CONTENT DATA
# =========================================================

content_df = joblib.load(
    "artifacts/models/content_df.pkl"
)


# =========================================================
# CONTENT BASED RECOMMENDATION
# =========================================================

def recommend_content_based(product_id, top_n=5):

    try:

        category = content_df[
            content_df['product_id'] == product_id
        ]['product_category_name'].values[0]

        similar_products = content_df[
            content_df['product_category_name'] == category
        ]

        similar_products = similar_products[
            similar_products['product_id'] != product_id
        ]

        recommendations = similar_products[
            'product_id'
        ].head(top_n).tolist()

        return recommendations

    except Exception:

        return []