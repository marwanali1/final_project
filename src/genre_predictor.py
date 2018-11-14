import datetime


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

    predictor = GenrePredictor()
    predictor.predict()
    # predictor.cross_validate()

    print('\nFINISH')
    print(datetime.datetime.time(datetime.datetime.now()))
