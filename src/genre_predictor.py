import datetime
import pandas as pd


class GenrePredictor:

    def __init__(self):
        self.training_data = None
        self.test_data = None
        self.__load_datasets()

    @staticmethod
    def create_datasets():
        """
        Used for creating training and test datasets from the original dataset
        """
        # Load the data set in the csv to a Pandas data frame
        msd_genre_df = pd.read_csv(filepath_or_buffer='data/audio_data/msd_genre_dataset.csv')

        training_set = None
        test_set = None
        for genre in msd_genre_df.genre.unique():
            genre_df = msd_genre_df.loc[msd_genre_df['genre'] == genre]
            training_sample = genre_df.sample(frac=0.25)
            genre_training_set = training_sample[
                ['genre', 'track_id', 'loudness', 'tempo', 'time_signature', 'key', 'mode', 'duration', 'avg_timbre1',
                 'avg_timbre2', 'avg_timbre3', 'avg_timbre4', 'avg_timbre5', 'avg_timbre6', 'avg_timbre7',
                 'avg_timbre8', 'avg_timbre9', 'avg_timbre10', 'avg_timbre11', 'avg_timbre12', 'var_timbre1',
                 'var_timbre2', 'var_timbre3', 'var_timbre4', 'var_timbre5', 'var_timbre6', 'var_timbre7',
                 'var_timbre8', 'var_timbre9', 'var_timbre10', 'var_timbre11', 'var_timbre12']]

            if training_set is None:
                training_set = genre_training_set
            else:
                training_set = pd.concat([training_set, genre_training_set])

            test_sample = pd.merge(genre_df, training_sample, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
            genre_test_set = test_sample[
                ['track_id', 'loudness', 'tempo', 'time_signature', 'key', 'mode', 'duration', 'avg_timbre1', 'avg_timbre2',
                 'avg_timbre3', 'avg_timbre4', 'avg_timbre5', 'avg_timbre6', 'avg_timbre7', 'avg_timbre8', 'avg_timbre9',
                 'avg_timbre10', 'avg_timbre11', 'avg_timbre12', 'var_timbre1', 'var_timbre2', 'var_timbre3', 'var_timbre4',
                 'var_timbre5', 'var_timbre6', 'var_timbre7', 'var_timbre8', 'var_timbre9', 'var_timbre10', 'var_timbre11',
                 'var_timbre12']]

            if test_set is None:
                test_set = genre_test_set
            else:
                test_set = pd.concat([test_set, genre_test_set])

        training_set.to_csv(path_or_buf='data/audio_data/training_data.csv', index=False)
        test_set.to_csv(path_or_buf='data/audio_data/test_data.csv', index=False)

    def __load_datasets(self):
        self.training_data = pd.read_csv(filepath_or_buffer='data/audio_data/training_data.csv')
        self.test_data = pd.read_csv(filepath_or_buffer='data/audio_data/test_data.csv')

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
