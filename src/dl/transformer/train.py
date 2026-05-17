# pyright: reportMissingImports=false

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from datasets import Dataset

from transformers import (

    Trainer,

    TrainingArguments

)

from src.dl.transformer.model import (
    load_bert_model
)

from src.dl.transformer.tokenizer import (
    load_tokenizer
)

from src.dl.transformer.utils import (
    compute_metrics
)


# =========================================================
# DEVICE
# =========================================================

import torch

device = torch.device(

    'cuda' if torch.cuda.is_available()
    else 'cpu'

)


# =========================================================
# SENTIMENT LABELS
# =========================================================

def create_sentiment(score):

    if score >= 4:
        return 2

    elif score == 3:
        return 1

    else:
        return 0


# =========================================================
# TOKENIZATION
# =========================================================

tokenizer = load_tokenizer()


def tokenize_function(examples):

    return tokenizer(

        examples['text'],

        padding='max_length',

        truncation=True,

        max_length=128

    )


# =========================================================
# TRAIN BERT
# =========================================================

def train_bert():

    print("\n=================================================")
    print("BERT TRAINING STARTED")
    print("=================================================\n")

    DATA_PATH = (
        "data/raw/order_reviews.csv"
    )

    df = pd.read_csv(DATA_PATH)

    nlp_df = df[[

        'review_comment_message',

        'review_score'

    ]].dropna()

    nlp_df['sentiment'] = nlp_df[
        'review_score'
    ].apply(create_sentiment)

    nlp_df = nlp_df.rename(columns={

        'review_comment_message': 'text',

        'sentiment': 'label'

    })

    nlp_df = nlp_df[[
        'text',
        'label'
    ]]

    # =====================================================
    # SAMPLE
    # =====================================================

    nlp_df = nlp_df.sample(

        n=min(5000, len(nlp_df)),

        random_state=42

    )

    # =====================================================
    # SPLIT
    # =====================================================

    train_df, test_df = train_test_split(

        nlp_df,

        test_size=0.2,

        random_state=42,

        stratify=nlp_df['label']

    )

    # =====================================================
    # HF DATASET
    # =====================================================

    train_dataset = Dataset.from_pandas(
        train_df
    )

    test_dataset = Dataset.from_pandas(
        test_df
    )

    # =====================================================
    # TOKENIZE
    # =====================================================

    train_dataset = train_dataset.map(

        tokenize_function,

        batched=True

    )

    test_dataset = test_dataset.map(

        tokenize_function,

        batched=True

    )

    # =====================================================
    # FORMAT
    # =====================================================

    train_dataset.set_format(

        type='torch',

        columns=[

            'input_ids',

            'attention_mask',

            'label'

        ]
    )

    test_dataset.set_format(

        type='torch',

        columns=[

            'input_ids',

            'attention_mask',

            'label'

        ]
    )

    # =====================================================
    # MODEL
    # =====================================================

    model = load_bert_model()

    model.to(device)

    # =====================================================
    # TRAINING ARGS
    # =====================================================

    training_args = TrainingArguments(

        output_dir='./bert_results',

        do_train=True,

        do_eval=True,

        learning_rate=2e-5,

        per_device_train_batch_size=4,

        per_device_eval_batch_size=4,

        num_train_epochs=1,

        weight_decay=0.01,

        logging_dir='./logs',

        logging_steps=100,

        save_steps=500,

        report_to=[]

    )

    # =====================================================
    # TRAINER
    # =====================================================

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=test_dataset,

        compute_metrics=compute_metrics

    )

    # =====================================================
    # TRAIN
    # =====================================================

    trainer.train()

    # =====================================================
    # SAVE MODEL
    # =====================================================

    os.makedirs(
        "artifacts/models",
        exist_ok=True
    )

    model.save_pretrained(
        "artifacts/models/bert_sentiment_model"
    )

    tokenizer.save_pretrained(
        "artifacts/models/bert_sentiment_model"
    )

    print("\n=================================================")
    print("BERT MODEL SAVED")
    print("=================================================\n")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    train_bert()