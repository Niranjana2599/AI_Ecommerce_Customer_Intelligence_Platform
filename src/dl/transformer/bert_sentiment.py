# pyright: reportMissingImports=false

import torch

from transformers import (

    BertTokenizer,

    BertForSequenceClassification

)


# =========================================================
# LOAD MODEL
# =========================================================

from src.utils.path_config import MODELS_DIR

model_path = MODELS_DIR / "bert_sentiment_model"

tokenizer = BertTokenizer.from_pretrained(
    model_path
)

model = BertForSequenceClassification.from_pretrained(
    model_path
)

model.eval()


# =========================================================
# LABEL MAP
# =========================================================

label_map = {

    0: "Negative",

    1: "Neutral",

    2: "Positive"

}


# =========================================================
# SENTIMENT PREDICTION
# =========================================================

def predict_sentiment(text):

    inputs = tokenizer(

        text,

        return_tensors='pt',

        truncation=True,

        padding=True,

        max_length=128

    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(

            outputs.logits,

            dim=1

        )

        prediction = torch.argmax(

            probabilities,

            dim=1

        ).item()

        confidence = probabilities[
            0
        ][prediction].item()

    return {

        "text": text,

        "sentiment": label_map[prediction],

        "confidence": round(confidence, 4)

    }