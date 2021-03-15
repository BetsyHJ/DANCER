import numpy as np
import pandas as pd
import os

class Dataset(object):
    def __init__(self, config, task='OIPT'):
        # np.random.seed(2021)
        path = config['data.input.path']
        dataset = config['data.input.dataset']
        self.path = path
        self.dataset = dataset
        # load data
        self.train_full = self._load_ratings(path + dataset + '/train.csv')
        self.test = self._load_ratings(path + dataset + '/test.csv')
        # # for task OPPT, we do randomly training-test splitting
        if task == 'OPPT':
            self.resplitting_random()
        
        # # add the predicted p(o) for task 2
        if task == 'OPPT': # "Observed user preference prediction task", add the predicted p(o) for task 2
            self.train_full = self.merge_predOP(self.train_full)
            self.test = self.merge_predOP(self.test)
            print("Load predOP (P(O)) done.")

        self.n_items = max(self.train_full['ItemId']) + 1
        self.n_users = max(self.train_full['UserId']) + 1
        

        # filter the items in the test but not in the training set
        self.test = self.test[self.test['ItemId'] < self.n_items]
        
        # get itemage
        self.item_birthdate = self._get_item_birthdate()

        self.period_type = config['data.itemage.type'].lower()
        if 'data.itemage.max' in config:
            self.n_periods = int(config['data.itemage.max'])
        if self.period_type == 'year':
            if 'data.itemage.max' not in config:
                self.n_periods = 20
        elif self.period_type == 'month':
            if 'data.itemage.max' not in config:
                self.n_periods = 36
        self.max_period = self.n_periods - 1
        self.n_bins = None
        train_itemage = self.get_itemage(self.train_full['ItemId'], self.train_full['timestamp'])
        self.n_periods = min(self.n_periods-1, max(train_itemage)) + 1
        self.train_full['ItemAge'] = train_itemage
        # # del the items only appearing in training set
        n_test = len(self.test)
        self.test = self.test.drop(self.test[~self.test['ItemId'].isin(self.train_full['ItemId'].unique())].index)
        print("Drop the items appearing in training set but not in test. Nr %d" % (n_test - len(self.test)))
        # exit(0)
        test_itemage = self.get_itemage(self.test['ItemId'], self.test['timestamp'])
        self.test['ItemAge'] = test_itemage
        if self.n_bins is not None:
            self.n_periods = self.n_bins
        
        # # get train and valid set
        # df_train, df_valid = [], []
        # for u, group in self.train_full.groupby(['UserId']):
        #     sorted_ratings = group.sort_values('timestamp', ascending=True)
        #     # # leave-one-out
        #     # df_train.append(sorted_ratings[:-1])
        #     # df_valid.append(sorted_ratings[-1:])
        #     # # user center, time-dependent, proportion
        #     n_valid = int(0.25 * len(sorted_ratings))
        #     df_train.append(sorted_ratings[:-n_valid])
        #     df_valid.append(sorted_ratings[-n_valid:])
        # self.train = pd.concat(df_train, axis=0)
        # self.valid = pd.concat(df_valid, axis=0)

        # basic info
        print("(#users : %d, #items : %d, period_type : %s, n_periods : %d)" % (self.n_users, self.n_items, self.period_type, self.n_periods))
        # self.train_full.to_csv('train_random-del.csv', index=False)
        # self.test.to_csv('test_random-del.csv', index=False)
        # exit(0)

    def _load_ratings(self, ratingfile, sep = ','):
        ratings = pd.read_csv(ratingfile, sep=sep, header=0)
        # ratings.columns = ['UserId', 'ItemId', 'rating', 'timestamp', 'ctr']
        return ratings

    def _get_item_birthdate(self):
        item_birthdate = np.zeros(self.n_items, dtype='int64')
        d = pd.concat([self.train_full, self.test], ignore_index=True)
        for i, group in d.groupby(['ItemId']):
            item_birthdate[i] = min(group['timestamp'].to_list())
        return item_birthdate
    
    def get_itemage(self, items, timestamp, item_birthdate = None, del_young_items=False):
        unit = self.period_type
        # max_period = self.n_periods - 1
        max_period = self.max_period
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
            period_second *= 365
        # self.period_second = period_second
        if item_birthdate is not None:
            item_pt = item_birthdate[items]
            itemage = ((timestamp - item_pt) * 1.0 / period_second).int().clip(0, max_period)
            # if del_young_items:
            #     itemage[(timestamp - item_pt) < 0] = -1
        else:
            item_pt = self.item_birthdate[items]
            # print(((timestamp - item_pt) * 1.0 / period_second))
            # exit(0)
            itemage = ((timestamp - item_pt) * 1.0 / period_second).astype(int).clip(0, max_period)
            # print(np.unique(itemage, return_counts=True))
            # if del_young_items:
            #     itemage[(timestamp - item_pt) < 0] = -1
            assert len(itemage) == len(items)
        
        # # # use bins to map the itemage
        # print('max_period:', max_period)
        # if self.dataset == 'ml-latest-small':
        #     itemage_ = np.copy(itemage)
        #     bins = [-1] + [0, 2, 4, 7, 10, 14, 20]
        #     print("---------- Using Bins: ", bins[1:], "----------")
        #     replaces = np.arange(len(bins) - 1)
        #     for bidx in range(1, len(bins)):
        #         b = list(range(bins[bidx-1]+1, bins[bidx]+1))
        #         itemage_[itemage.isin(b)] = replaces[bidx - 1]
        #     self.n_bins = min(self.n_periods, len(replaces))
        #     # print(np.unique(itemage, return_counts=True))
        #     # print(np.unique(itemage_, return_counts=True), '\n')
        #     # exit(0)
        #     return itemage_
        return itemage

    def merge_predOP(self, train):
        if self.dataset == 'ml-latest-small':
            filename = self.path + self.dataset + '/predOP_tmf_fast_v.csv'
        elif self.dataset == 'ml-100k':
            filename = self.path + self.dataset + '/predOP_mf.csv'
        else:
            raise NotImplementedError("Only implement for ml-100k or ml-latest-small")
        predOP = self._load_ratings(filename)
        predOP = predOP[['UserId', 'ItemId', 'rating', 'timestamp', 'predOP']]
        del_columns = list(set(predOP.columns) - set(train.columns) - set(['predOP']))
        # print(del_columns)
        predOP.drop(columns=del_columns)
        # print("the remained columns of predOP is", predOP.columns)
        # train_ = train.copy()
        # output = train_.append(predOP, ignore_index=True, sort=False)
        # output.dropna(axis=0, how='any') #drop all rows that have any NaN values
        # output = pd.concat([train, predOP], axis=1, join='inner')
        # output = output[output.columns].T.drop_duplicates().T
        output = pd.merge(train, predOP, how='inner', on=['UserId', 'ItemId', 'rating', 'timestamp'])
        # print("After concat", output.columns, '\n')
        # print(output['UserId'])
        # print(len(output), len(train))
        # assert len(output) == len(train)
        return output

    def resplitting_random(self, ratio = 0.25):
        if os.path.exists(self.path + self.dataset + '/train_random.csv'):
            self.train_full = self._load_ratings(self.path + self.dataset + '/train_random.csv')
            self.test = self._load_ratings(self.path + self.dataset + '/test_random.csv')
            # only keep ['UserId', 'ItemId', 'rating', 'timestamp']
            cols = ['UserId', 'ItemId', 'rating', 'timestamp']
            self.train_full = self.train_full[cols]
            self.test = self.test[cols]
            print("Loading randomly-splitted training and test set. Done.")
            return
        print("Generate the random-splitted training/test set.")
        # # combine train_full and test as a whole dataset
        df = pd.concat([self.train_full, self.test], ignore_index=True)
        # # random splitting by user
        df_train, df_test = [], []
        np.random.seed(2021)
        for (u, group) in df.groupby(['UserId']):
            group_ = group.sample(frac=1)
            n_test = int(ratio * len(group_))
            df_train.append(group_[:-n_test])
            df_test.append(group_[-n_test:])

        self.train_full = pd.concat(df_train, axis=0)
        self.test = pd.concat(df_test, axis=0)
        # only keep ['UserId', 'ItemId', 'rating', 'timestamp']
        cols = ['UserId', 'ItemId', 'rating', 'timestamp']
        self.train_full = self.train_full[cols]
        self.test = self.test[cols]
        self.train_full.to_csv(self.path + self.dataset + '/train_random.csv', index=False)
        self.test.to_csv(self.path + self.dataset + '/test_random.csv', index=False)
        print("Saving randomly-splitted dataset. Done.")

