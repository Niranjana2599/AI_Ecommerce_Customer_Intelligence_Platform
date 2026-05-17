import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from src.nlp.preprocessing import clean_text

from src.utils.path_config import MODELS_DIR

# =========================================================
# TRAIN TFIDF
# =========================================================


def train_tfidf_vectorizer():

    DATA_PATH = "data/raw/order_reviews.csv"

    df = pd.read_csv(DATA_PATH)

    nlp_df = df[[
        'review_comment_message'
    ]].dropna()

    nlp_df['cleaned_review'] = nlp_df[
        'review_comment_message'
    ].apply(clean_text)

    vectorizer = TfidfVectorizer(
        max_features=5000
    )

    tfidf_matrix = vectorizer.fit_transform(
        nlp_df['cleaned_review']
    )

    joblib.dump(
    vectorizer,
    MODELS_DIR / "tfidf_vectorizer.pkl")

    print("\nTFIDF Vectorizer Saved Successfully")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    train_tfidf_vectorizer()