class FeatureEngineering:

    def __init__(self, dataframe):

        self.df = dataframe

    def transform(self):

        self.df.columns = [

            col.lower() for col in self.df.columns
        ]

        return self.df