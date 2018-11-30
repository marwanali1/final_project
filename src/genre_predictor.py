import pandas as pd

from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import KFold


class GenrePredictor:

    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.__load_datasets()

        self.class_weights = {
            'classic pop and rock': 1,
            'classical': 2,
            'dance and electronica': 2,
            'folk': 1,
            'hip-hop': 3,
            'jazz and blues': 2,
            'metal': 2,
            'pop': 2,
            'punk': 2,
            'soul and reggae': 2
        }

    @staticmethod
    def create_datasets():
        """
        Used for creating training and test datasets from the original dataset
        """
        # Load the data set in the csv to a Pandas data frame
        msd_genre_df = pd.read_csv(filepath_or_buffer='data/audio_data/msd_genre_dataset.csv')

        rel_features = ['genre', 'track_id', 'loudness', 'tempo', 'time_signature', 'key', 'mode', 'duration',
                        'avg_timbre1',
                        'avg_timbre2', 'avg_timbre3', 'avg_timbre4', 'avg_timbre5', 'avg_timbre6', 'avg_timbre7',
                        'avg_timbre8', 'avg_timbre9', 'avg_timbre10', 'avg_timbre11', 'avg_timbre12', 'var_timbre1',
                        'var_timbre2', 'var_timbre3', 'var_timbre4', 'var_timbre5', 'var_timbre6', 'var_timbre7',
                        'var_timbre8', 'var_timbre9', 'var_timbre10', 'var_timbre11', 'var_timbre12']

        training_set = None
        test_set = None
        for genre in msd_genre_df.genre.unique():
            genre_df = msd_genre_df.loc[msd_genre_df['genre'] == genre]
            training_sample = genre_df.sample(frac=0.30)
            genre_training_set = training_sample[rel_features]

            if training_set is None:
                training_set = genre_training_set
            else:
                training_set = pd.concat([training_set, genre_training_set])

            test_sample = pd.merge(genre_df, training_sample, indicator=True, how='outer').query(
                '_merge=="left_only"').drop('_merge', axis=1)
            genre_test_set = test_sample[rel_features]

            if test_set is None:
                test_set = genre_test_set
            else:
                test_set = pd.concat([test_set, genre_test_set])

        training_set.to_csv(path_or_buf='data/audio_data/train_data.csv', index=False)
        test_set.to_csv(path_or_buf='data/audio_data/test_data.csv', index=False)

    def __load_datasets(self):
        self.train_data = pd.read_csv(filepath_or_buffer='data/audio_data/train_data.csv')
        self.test_data = pd.read_csv(filepath_or_buffer='data/audio_data/test_data.csv')

    def __evaluate(self, predictions, true_genres=None):
        if true_genres is None:
            true_genres = pd.DataFrame(self.test_data['genre']).values.reshape(-1, ).tolist()

        # Bad metric for this dataset since it's imbalanced
        accuracy = accuracy_score(true_genres, predictions)
        print("\nAccuracy score: {}".format(accuracy))

        # A measure of a classifiers exactness
        precision = precision_score(true_genres, predictions, average='micro')
        print("Precision Score: {}".format(precision))

        # A measure of a classifiers completeness
        recall = recall_score(true_genres, predictions, average='micro')
        print("Recall Score: {}".format(recall))

        # A weighted average of precision and recall
        f1 = f1_score(true_genres, predictions, average='micro')
        print("F1 Score: {}".format(f1))

    def cross_validate(self):
        features = ['loudness', 'tempo', 'time_signature', 'key', 'mode', 'duration', 'avg_timbre1', 'avg_timbre2',
                    'avg_timbre3', 'avg_timbre4', 'avg_timbre5', 'avg_timbre6', 'avg_timbre7', 'avg_timbre8',
                    'avg_timbre9',
                    'avg_timbre10', 'avg_timbre11', 'avg_timbre12', 'var_timbre1', 'var_timbre2', 'var_timbre3',
                    'var_timbre4',
                    'var_timbre5', 'var_timbre6', 'var_timbre7', 'var_timbre8', 'var_timbre9', 'var_timbre10',
                    'var_timbre11',
                    'var_timbre12']

        df = self.train_data[features]
        kf = KFold(n_splits=3, shuffle=True, random_state=2)
        splits = kf.split(df)

        classifier = LogisticRegression(penalty='l1', solver='liblinear', multi_class='auto', max_iter=300)
        for result in splits:
            train = df.iloc[result[0]]
            test = df.iloc[result[1]]

            train_genres = pd.DataFrame(self.train_data.iloc[train.index.values].genre).values.reshape(-1, ).tolist()
            true_test_genres = pd.DataFrame(self.train_data.iloc[test.index.values].genre).values.reshape(-1, ).tolist()

            classifier.fit(train, train_genres)
            predictions = classifier.predict(test)

            self.__evaluate(predictions=predictions, true_genres=true_test_genres)

    def predict(self):
        features = ['loudness', 'tempo', 'time_signature', 'key', 'mode', 'duration', 'avg_timbre1', 'avg_timbre2',
                    'avg_timbre3', 'avg_timbre4', 'avg_timbre5', 'avg_timbre6', 'avg_timbre7', 'avg_timbre8',
                    'avg_timbre9',
                    'avg_timbre10', 'avg_timbre11', 'avg_timbre12', 'var_timbre1', 'var_timbre2', 'var_timbre3',
                    'var_timbre4',
                    'var_timbre5', 'var_timbre6', 'var_timbre7', 'var_timbre8', 'var_timbre9', 'var_timbre10',
                    'var_timbre11',
                    'var_timbre12']

        unlabled_train_set = self.train_data[features]
        unlabled_test_set = self.test_data[features]

        genres = pd.DataFrame(self.train_data['genre']).values.reshape(-1, ).tolist()

        classifier = LogisticRegression(penalty='l1', class_weight=self.class_weights, solver='liblinear', multi_class='auto')
        classifier.fit(unlabled_train_set, genres)
        predictions = classifier.predict(unlabled_test_set)

        self.__evaluate(predictions=predictions)


if __name__ == "__main__":
    print("START TIME: {}".format(datetime.time(datetime.now())))

    # GenrePredictor.create_datasets()

    predictor = GenrePredictor()
    predictor.predict()
    # predictor.cross_validate()

    print("\nFINISH TIME: {}".format(datetime.time(datetime.now())))
