import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

class Regressor(BaseEstimator):
    def __init__(self):
        self.model =  KNeighborsRegressor(n_neighbors=20)
        self.sc = StandardScaler()

    def fit(self, X, y):
        y_to_fit = pd.DataFrame(y[:, 0])
        y_to_fit['authorId'] = X['authorId'].values
        y_to_fit['beerId'] = X['beerId'].values
        self.sc.fit(X)
        self.model.fit(self.sc.transform(X), y_to_fit)


    def predict(self, X):
        return(predict_rating(self.model, X))

def predict_rating(model, df):


    ### sizes
    closestN = model.kneighbors(df)[1]
    df = pd.DataFrame(df).rename(columns={0: 'authorId', 1: 'beerId', 5: 'mean_rating'})
    n = df.shape[0]
    y_pred = np.ndarray((n, 2   ))
    y = model._y
    y = pd.DataFrame(y).rename(columns={0: 'rating', 1: 'authorId', 2: 'beerId'})
    for i in range(n):
        if i % 1000 == 0 :
            print (i)
        #### We get the id of the beer and it's mean rating by user
        the_i_beer = df.iloc[i]['beerId']
        mean_rate = df.iloc[i]['mean_rating']
        ##### we get the list of the closest neighbours for each prediction

        neibb_i = np.unique(y.iloc[closestN[i]]['authorId'].values)


        ### We get their id
        df_i = y.loc[y['authorId'].isin(neibb_i)]

        ### Faire i loc sur 3ieme colonne pour trouver
        ### We compute there correlation to get weights
        corr_i = df_i.pivot_table(index='beerId', columns='authorId', values='rating', dropna=False).corr()
        weights = corr_i.fillna(0).iloc[0].values

        ### We look for the rating by those people of the beer
        df_with_the_beer = y[y.authorId.isin(neibb_i[1:])][y.beerId == the_i_beer] \
            [["authorId", "rating"]]

        rate = 0
        weights_sum = 0
        for j in df_with_the_beer['authorId']:
            id_i_rating = df_with_the_beer[df_with_the_beer.authorId == j]['rating'].values[0]
            id_i_pos = np.where(neibb_i == j)
            rate += weights[id_i_pos] * id_i_rating
            weights_sum += weights[id_i_pos]
        ### If they didnt rate we give the average grade, else we give the average weighted rate of it's neighbours
        if weights_sum == 0:
            y_pred[i,:] = [mean_rate, mean_rate]
        else:
            y_pred[i,: ] = [np.abs(rate) / np.abs(weights_sum), mean_rate]
    return y_pred
