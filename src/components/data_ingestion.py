import pandas as pd

class DataIngestion:

    def __init__(self, path):

        self.path = path

    def load_data(self):

        df = pd.read_csv(self.path)

        return df