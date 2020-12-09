import numpy as np
import pandas as pd

class Dataset(object):
    def __init__(self, config):
        path = config['data.input.path']
        dataset = config['data.input.dataset']
        
        # load data
        self.train_full = self._load_ratings(path + dataset + '/train.csv')
        self.test = self._load_ratings(path + dataset + '/test.csv')

        # time info
        max_month = 5 * 12 - 1 # 5 years
        self.train_full['ItemAge_month'] = self.train_full['ItemAge_month'].clip(0, max_month)
        self.test['ItemAge_month'] = self.test['ItemAge_month'].clip(0, max_month)
        self.n_months = max(self.train_full['ItemAge_month']) + 1
        
        # get train and valid set
        df_train, df_valid = [], []
        for u, group in self.train_full.groupby(['UserId']):
            sorted_ratings = group.sort_values('timestamp', ascending=True)
            # # leave-one-out
            # df_train.append(sorted_ratings[:-1])
            # df_valid.append(sorted_ratings[-1:])
            # # user center, time-dependent, proportion
            n_valid = int(0.25 * len(sorted_ratings))
            df_train.append(sorted_ratings[:-n_valid])
            df_valid.append(sorted_ratings[-n_valid:])

        self.train = pd.concat(df_train, axis=0)
        self.valid = pd.concat(df_valid, axis=0)

        # basic info
        self.n_items = max(self.train_full['ItemId']) + 1
        self.n_users = max(self.train_full['UserId']) + 1

        # filter the items in the test but not in the training set
        self.test = self.test[self.test['ItemId'] < self.n_items]
        

    def _load_ratings(self, ratingfile, sep = ','):
        ratings = pd.read_csv(ratingfile, sep=sep, header=0)
        # ratings.columns = ['UserId', 'ItemId', 'rating', 'timestamp', 'ctr']
        return ratings

    def _get_item_birthdate(self):
        item_birthdate = np.zeros(self.n_items, dtype='int64')
        for i, group in self.train_full.groupby(['ItemId']):
            item_birthdate[i] = min(group['timestamp'].to_list())
        return item_birthdate
