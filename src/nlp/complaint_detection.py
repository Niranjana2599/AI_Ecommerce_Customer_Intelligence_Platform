from textblob import TextBlob


# =========================================================
# COMPLAINT DETECTION
# =========================================================

def detect_complaint(text):

    polarity = TextBlob(text).sentiment.polarity

    if polarity < 0:

        return {

            "complaint_detected": True,

            "polarity": round(polarity, 4),

            "status": "Negative Customer Complaint"

        }

    return {

        "complaint_detected": False,

        "polarity": round(polarity, 4),

        "status": "No Complaint Detected"

    }


# =========================================================
# TESTING
# =========================================================

if __name__ == "__main__":

    sample_text = (
        "Delivery was extremely poor and disappointing"
    )

    result = detect_complaint(sample_text)

    print("\nComplaint Detection Result:\n")

    print(result)