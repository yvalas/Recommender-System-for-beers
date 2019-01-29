from scipy import constants
import pandas as pd


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y):

        return(self, X_df)

    def transform(self, X_df):
        return X_df[['authorId',	'beerId',	'styleId',	'brewerId',	'abv', 'mean_rating'	]]



