import numpy as np
import pandas as pd

class Dataset(object):
    def __init__(self, config):
        path = config['data.input.path']
        dataset = config['data.input.dataset']
        
        # load data
        self.train_full = self._load_ratings(path + dataset + '/train.csv')
        self.test = self._load_ratings(path + dataset + '/test.csv')
        self.n_items = max(self.train_full['ItemId']) + 1
        self.n_users = max(self.train_full['UserId']) + 1

        # filter the items in the test but not in the training set
        self.test = self.test[self.test['ItemId'] < self.n_items]
        
        # get itemage
        self.item_birthdate = self._get_item_birthdate()

        # self.period_type = 'year'
        # self.n_periods = 10
        self.period_type = 'month'
        self.n_periods = 12
        train_itemage = self.get_itemage(self.train_full['ItemId'], self.train_full['timestamp'])
        self.n_periods = min(self.n_periods-1, max(train_itemage)) + 1
        self.train_full['ItemAge'] = train_itemage
        test_itemage = self.get_itemage(self.test['ItemId'], self.test['timestamp'])
        self.test['ItemAge'] = test_itemage
        # print(min(self.test['ItemAge']))
        # assert min(test_itemage) < self.n_periods
        
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
        print("(#users : %d, #items : %d, period_type : %s, n_periods : %d)" % (self.n_users, self.n_items, self.period_type, self.n_periods))

    def _load_ratings(self, ratingfile, sep = ','):
        ratings = pd.read_csv(ratingfile, sep=sep, header=0)
        # ratings.columns = ['UserId', 'ItemId', 'rating', 'timestamp', 'ctr']
        return ratings

    def _get_item_birthdate(self):
        item_birthdate = np.zeros(self.n_items, dtype='int64')
        for i, group in self.train_full.groupby(['ItemId']):
            item_birthdate[i] = min(group['timestamp'].to_list())
        return item_birthdate
    
    def get_itemage(self, items, timestamp, item_birthdate = None):
        unit = self.period_type
        max_period = self.n_periods - 1
        '''
        # time info
        max_month = 5 * 12 - 1 # 5 years
        self.train_full['ItemAge_month'] = self.train_full['ItemAge_month'].clip(0, max_month)
        self.test['ItemAge_month'] = self.test['ItemAge_month'].clip(0, max_month)
        self.n_months = max(self.train_full['ItemAge_month']) + 1
        '''

        period_second = 24 * 60 * 60 # how many seconds in one day.
        if unit == 'month':
            period_second *= 30
        elif unit == 'year':
            period_second *= 30 * 356
        # self.period_second = period_second
        if item_birthdate is not None:
            item_pt = item_birthdate[items]
            itemage = ((timestamp - item_pt) * 1.0 / period_second).int().clip(0, max_period)
        else:
            item_pt = self.item_birthdate[items]
            itemage = ((timestamp - item_pt) * 1.0 / period_second).astype(int).clip(0, max_period)
            assert len(itemage) == len(items)
        return itemage

