import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.nlp.preprocessing import clean_text

from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.path_config import MODELS_DIR

# =========================================================
# TRAIN SENTIMENT MODEL
# =========================================================


def train_sentiment_model():

    print("\n=================================================")
    print("NLP SENTIMENT MODEL TRAINING STARTED")
    print("=================================================\n")

    DATA_PATH = "data/raw/order_reviews.csv"

    df = pd.read_csv(DATA_PATH)

    nlp_df = df[[
        'review_comment_message',
        'review_score'
    ]].dropna()

    # =====================================================
    # SENTIMENT LABEL
    # =====================================================

    def create_sentiment(score):

        if score >= 4:

            return 1

        return 0

    nlp_df['sentiment'] = nlp_df[
        'review_score'
    ].apply(create_sentiment)

    # =====================================================
    # CLEAN TEXT
    # =====================================================

    nlp_df['cleaned_review'] = nlp_df[
        'review_comment_message'
    ].apply(clean_text)

    # =====================================================
    # TFIDF
    # =====================================================

    vectorizer = TfidfVectorizer(
        max_features=5000
    )

    X = vectorizer.fit_transform(
        nlp_df['cleaned_review']
    )

    y = nlp_df['sentiment']

    # =====================================================
    # TRAIN TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # =====================================================
    # MODEL
    # =====================================================

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(X_train, y_train)

    # =====================================================
    # EVALUATION
    # =====================================================

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\nAccuracy:", accuracy)

    print("\nClassification Report\n")

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print("\nConfusion Matrix\n")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    # =====================================================
    # SAVE ARTIFACTS
    # =====================================================

    joblib.dump(
    model,
    MODELS_DIR / "logistic_sentiment_model.pkl")


    joblib.dump(
    vectorizer,
    MODELS_DIR / "tfidf_vectorizer.pkl")
    

    print("\n=================================================")
    print("NLP MODEL SAVED SUCCESSFULLY")
    print("=================================================\n")


# =========================================================
# PREDICT SENTIMENT
# =========================================================

model = joblib.load(
    MODELS_DIR / "logistic_sentiment_model.pkl"
)

vectorizer = joblib.load(
    MODELS_DIR / "tfidf_vectorizer.pkl"
)



def predict_sentiment(text):

    cleaned = clean_text(text)

    vectorized = vectorizer.transform([
        cleaned
    ])

    prediction = model.predict(vectorized)[0]

    probability = model.predict_proba(
        vectorized
    )[0][1]

    return {
        "prediction": int(prediction),
        "positive_probability": round(float(probability), 4)
    }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    train_sentiment_model()

    sample_text = "Product quality is amazing"

    result = predict_sentiment(sample_text)

    print(result)