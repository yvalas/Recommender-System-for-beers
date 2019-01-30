import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import ShuffleSplit
import numpy as np


from rampwf.score_types.base import BaseScoreType



problem_title = 'Recommender System for beers'
_target_column_names = ['rating', 'mean_rating']

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression(
    label_names=['rating', 'mean_rating'])
# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorRegressor()

class RMSE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='RMSE', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_true[:, 0] - y_pred[:, 0])))


class Res(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='residual dev', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return np.mean(np.abs(y_true[:, 0] - y_pred[:, 0])) / np.mean(np.abs(y_true[:, 1] - y_pred[:, 0]))



score_types = [
    RMSE(), Res()
]







def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_names].values
    X_df = data.drop('rating', axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)


def get_cv(X, y):

    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=57)
    return cv.split(X, y[:, 0])

