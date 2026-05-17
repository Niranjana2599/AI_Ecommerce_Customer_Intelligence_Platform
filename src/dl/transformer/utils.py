import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =========================================================
# COMPUTE METRICS
# =========================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        average='weighted'
    )

    recall = recall_score(
        labels,
        predictions,
        average='weighted'
    )

    f1 = f1_score(
        labels,
        predictions,
        average='weighted'
    )

    return {

        'accuracy': accuracy,

        'precision': precision,

        'recall': recall,

        'f1': f1

    }