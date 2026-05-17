import pandas as pd

from gensim.models import Word2Vec

from src.nlp.preprocessing import clean_text


# =========================================================
# TRAIN WORD2VEC
# =========================================================


def train_word2vec():

    DATA_PATH = "data/raw/order_reviews.csv"

    df = pd.read_csv(DATA_PATH)

    reviews = df[
        'review_comment_message'
    ].dropna()

    cleaned_reviews = reviews.apply(clean_text)

    tokenized_reviews = [
        text.split()
        for text in cleaned_reviews
    ]

    model = Word2Vec(
        sentences=tokenized_reviews,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4
    )

    model.save(
        "artifacts/models/word2vec.model"
    )

    print("\nWord2Vec Model Saved")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    train_word2vec()