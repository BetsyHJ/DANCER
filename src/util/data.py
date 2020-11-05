import numpy as np
import pandas as pd

class Dataset(object):
    def __init__(self, config):
        path = config['data.input.path']
        dataset = config['data.input.dataset']
        
        # load data
        self.train_full = self._load_ratings(path + dataset + '/train.csv')
        self.test = self._load_ratings(path + dataset + '/test.csv')
        # get train and valid set
        df_train, df_valid = [], []
        for u, group in self.train_full.groupby(['UserId']):
            sorted_ratings = group.sort_values('timestamp', ascending=True)
            df_train.append(sorted_ratings[:-1])
            df_valid.append(sorted_ratings[-1:])
        self.train = pd.concat(df_train, axis=0)
        self.valid = pd.concat(df_valid, axis=0)

        # basic info
        self.n_items = max(self.train_full['ItemId']) + 1
        self.n_users = max(self.train_full['UserId']) + 1

    def _load_ratings(self, ratingfile, sep = ','):
        ratings = pd.read_csv(ratingfile, sep=sep, header=0)
        ratings.columns = ['UserId', 'ItemId', 'rating', 'timestamp', 'ctr']
        return ratings

