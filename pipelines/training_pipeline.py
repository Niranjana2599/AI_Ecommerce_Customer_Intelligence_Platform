# =========================================================
# PATH SETUP
# =========================================================

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

# =========================================================
# IMPORTS
# =========================================================

import pandas as pd

from src.ml.churn.train import train_churn_model


# =========================================================
# LOAD DATA
# =========================================================

def run_training_pipeline():

    df = pd.read_csv('data/processed/churn_data.csv')

    train_churn_model(df)


if __name__ == '__main__':
    run_training_pipeline()