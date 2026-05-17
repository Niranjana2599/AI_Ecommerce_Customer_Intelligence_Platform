from src.dl.transformer.bert_sentiment import (
    predict_sentiment
)

from src.dl.transformer.review_classifier import (
    classify_review
)

from src.dl.transformer.summarizer import (
    summarize_review
)

from src.dl.transformer.embeddings import (
    generate_embeddings
)


# =========================================================
# ALL NLP PREDICTIONS
# =========================================================

def run_all_predictions(text):

    sentiment = predict_sentiment(text)

    category = classify_review(text)

    summary = summarize_review(text)

    embeddings = generate_embeddings(text)

    return {

        "sentiment": sentiment,

        "category": category,

        "summary": summary,

        "embedding_dimension":

            embeddings['embedding_dimension']

    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    sample = (
        "Delivery was delayed but product quality is excellent"
    )

    result = run_all_predictions(sample)

    print(result)