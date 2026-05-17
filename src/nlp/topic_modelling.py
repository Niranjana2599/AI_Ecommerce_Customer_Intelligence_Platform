import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.feature_extraction.text import CountVectorizer

from src.nlp.preprocessing import clean_text


# =========================================================
# TOPIC MODELLING
# =========================================================


def run_topic_modelling():

    DATA_PATH = "data/raw/order_reviews.csv"

    df = pd.read_csv(DATA_PATH)

    reviews = df[
        'review_comment_message'
    ].dropna()

    cleaned_reviews = reviews.apply(clean_text)

    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        stop_words='english'
    )

    dtm = vectorizer.fit_transform(
        cleaned_reviews
    )

    lda = LatentDirichletAllocation(
        n_components=5,
        random_state=42
    )

    lda.fit(dtm)

    print("\n=================================================")
    print("TOPIC MODELLING")
    print("=================================================\n")

    for index, topic in enumerate(lda.components_):

        print(f"\nTopic #{index}\n")

        print([
            vectorizer.get_feature_names_out()[i]
            for i in topic.argsort()[-10:]
        ])


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    run_topic_modelling()