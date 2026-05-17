class DataCleaning:

    def __init__(self, dataframe):

        self.df = dataframe

    def clean_data(self):

        self.df.drop_duplicates(inplace=True)

        self.df.dropna(inplace=True)

        return self.df