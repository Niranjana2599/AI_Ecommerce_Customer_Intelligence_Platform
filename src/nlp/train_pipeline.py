from src.nlp.sentiment_analysis import (
    train_sentiment_model
)

from src.nlp.tfidf_vectorizer import (
    train_tfidf_vectorizer
)

from src.nlp.word2vec import (
    train_word2vec
)


# =========================================================
# TRAIN NLP PIPELINE
# =========================================================

def run_nlp_pipeline():

    print("\n=================================================")
    print("NLP TRAINING PIPELINE STARTED")
    print("=================================================\n")

    train_tfidf_vectorizer()

    train_sentiment_model()

    train_word2vec()

    print("\n=================================================")
    print("NLP TRAINING PIPELINE COMPLETED")
    print("=================================================\n")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    run_nlp_pipeline()