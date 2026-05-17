from src.nlp.sentiment_analysis import (
    predict_sentiment
)

from src.nlp.summarization import (
    summarize_review
)

from src.nlp.complaint_detection import (
    detect_complaint
)


# =========================================================
# NLP PREDICTION PIPELINE
# =========================================================

def run_prediction_pipeline(text):

    sentiment = predict_sentiment(text)

    summary = summarize_review(text)

    complaint = detect_complaint(text)

    return {

        "sentiment": sentiment,

        "summary": summary,

        "complaint": complaint

    }


# =========================================================
# TESTING
# =========================================================

if __name__ == "__main__":

    sample_text = (
        "Delivery was late but product quality was good"
    )

    result = run_prediction_pipeline(sample_text)

    print("\nPrediction Pipeline Result:\n")

    print(result)