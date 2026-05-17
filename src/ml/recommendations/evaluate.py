import joblib


# =========================================================
# LOAD ARTIFACTS
# =========================================================

product_similarity_df = joblib.load(
    "artifacts/models/product_similarity.pkl"
)


# =========================================================
# EVALUATE COVERAGE
# =========================================================


def evaluate_recommendation_system():

    recommended_counts = []

    sample_products = list(
        product_similarity_df.index[:100]
    )

    for product_id in sample_products:

        similar_products = product_similarity_df[
            product_id
        ].sort_values(ascending=False)

        similar_products = similar_products.iloc[1:6]

        recommended_counts.append(
            len(similar_products)
        )

    avg_recommendations = sum(
        recommended_counts
    ) / len(recommended_counts)

    print("\n=================================================")
    print("RECOMMENDATION SYSTEM EVALUATION")
    print("=================================================\n")

    print(
        f"Average Recommendations Per Product: {avg_recommendations:.2f}"
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    evaluate_recommendation_system()