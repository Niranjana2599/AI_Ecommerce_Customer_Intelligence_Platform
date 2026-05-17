import joblib


# =========================================================
# LOAD PICKLE MODEL
# =========================================================

def load_pickle_model(path):

    return joblib.load(path)


# =========================================================
# LOAD KERAS MODEL
# =========================================================

def load_keras_model(path):

    import tensorflow as tf

    return tf.keras.models.load_model(path)