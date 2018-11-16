import datetime
import pandas as pd


class GenrePredictor:

    def __init__(self):
        self.training_data = None
        self.test_data = None

        self.__parse_training_data()
        self.__parse_test_data()

    def __clean_line(self, line):
        pass

    def __parse_training_data(self):
        pass

    def __parse_test_data(self):
        pass

    def cross_validate(self):
        pass

    def predict(self):
        pass


if __name__ == "__main__":
    print("START")
    print(datetime.datetime.time(datetime.datetime.now()))

    # Load the data set in the csv to a Pandas data frame
    msd_genre_df = pd.read_csv('data/msd_genre_dataset.csv')

    # Print all the columns in the data set
    col_names = list(msd_genre_df.columns.values)
    print(col_names)

    # Print the first row of the data set
    print(msd_genre_df.iloc[0])

    # predictor = GenrePredictor()
    # predictor.predict()
    # predictor.cross_validate()

    print('\nFINISH')
    print(datetime.datetime.time(datetime.datetime.now()))
