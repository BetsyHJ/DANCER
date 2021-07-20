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
        assert (task.upper() != 'TART') or (dataset in ['simulation', 'simulation2', 'simulation3','fully-synthetic']) # make sure when the task is TART, we use simulated dataset
        assert (task.upper() == 'TART') or (dataset not in ['simulation', 'simulation2', 'simulation3','fully-synthetic']) # make sure simulated dataset be used only when task is TART

        # # load data
        if task.upper() == 'TART':
            # train columns: [UserId,ItemId,itemage,rating,predOP]
            self.train = self._load_ratings(path + dataset + '/train.csv')
            self.valid = self._load_ratings(path + dataset + '/valid.csv')
            # test columns: [UserId,ItemId,rating,itemage]
            self.test = self._load_ratings(path + dataset + '/test.csv')
            self.test['itemage_copy'] = np.copy(self.test['itemage'].values) # for getting the original itemage
            self.n_users = max(self.train['UserId']) + 1 #max(max(self.train['UserId']), max(self.valid['UserId']))
            self.n_items = max(self.train['ItemId']) + 1 #max(max(self.train['ItemId']), max(self.valid['ItemId']))
            self.n_periods = max(max(self.train['itemage']), max(self.valid['itemage']), max(self.test['itemage'])) + 1

            ''' use estimated propensities / predOP '''
            # self.estimated_predOP_replace(mode='b2_i', static=False)
            # self.estimated_predOP_replace(mode='b3', static=False)
            # self.estimated_predOP_replace(mode='b4', static=False)
            # self.estimated_predOP_replace(mode='mf', static=False)
            # self.estimated_predOP_replace(mode='tmtf', static=False)
    
            # map the itemage into the bins:
            if 'simulation' in dataset:
                self.train['itemage'], _ = self.years2bins(self.train['itemage'])
                self.valid['itemage'], _ = self.years2bins(self.valid['itemage'])
                self.test['itemage'], n_bins = self.years2bins(self.test['itemage'])
                self.n_periods = n_bins
            # print(self.train['itemage'].values[800:850])
            # del items in test but not in train
            n_test = len(self.test)
            self.test = self.test[self.test['ItemId'].isin(self.train['ItemId'].unique())]
            print("Simulated test: %d interactions whose items do not appear in the training set. And thus Nr. testset is %d." % (n_test - len(self.test), len(self.test)))
            print("(#users : %d, #items : %d, period_type : %s, n_periods : %d)" % (self.n_users, self.n_items, config['data.itemage.type'].lower(), self.n_periods))
            return

        self.task = task
        # set the period info
        self.period_type = config['data.itemage.type'].lower()
        if 'data.itemage.max' in config:
            self.n_periods = int(config['data.itemage.max'])
        if self.period_type == 'year':
            if 'data.itemage.max' not in config:
                self.n_periods = 20
        elif self.period_type == 'month':
            if 'data.itemage.max' not in config:
                self.n_periods = 36

        # self.train_full = self._load_ratings(path + dataset + '/train.csv')
        # self.test = self._load_ratings(path + dataset + '/test.csv')

        # OIPT, do random splitting
        if (task == 'OIPT') and (config['splitting'] == 'random'):
            self.resplitting_random_OIPT()
            # self.resplitting_random_OIPT3(ratio=0.1)
            print("(#users : %d, #items : %d, period_type : %s, n_periods : %d)" % (self.n_users, self.n_items, self.period_type, self.n_periods))
            return

        self.train_full = self._load_ratings(path + dataset + '/train.csv')
        self.test = self._load_ratings(path + dataset + '/test.csv')

        # self._get_item_birthdate() # put it here to generate ./data/simulation/item_birthdate.csv
        # # for task OPPT, we do randomly training-test splitting
        if task == 'OPPT':
            self.resplitting_random()
        
        # # add the predicted p(o) for task 2
        if (task == 'OPPT') and config['debiasing']: # "Observed user preference prediction task", add the predicted p(o) for task 2
            self.train = self.merge_predOP(self.train)
            self.valid = self.merge_predOP(self.valid)
            self.test = self.merge_predOP(self.test)
            print("Load predOP (P(O)) done.")

        self.n_items = max(self.train_full['ItemId']) + 1
        self.n_users = max(self.train_full['UserId']) + 1
        

        # filter the items in the test but not in the training set
        self.test = self.test[self.test['ItemId'] < self.n_items]
        
        # get itemage
        self.item_birthdate = self._get_item_birthdate()

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

        # # get training and valid observation data for model learning
        # OIPT, do time-based splitting, generate train_interactions and valid_interactions with 9:1 splitting
        if (task == 'OIPT') and (config['splitting'] != 'random'):
            self.interation_data_time_OIPT(self.train_full)

        # basic info
        print("(#users : %d, #items : %d, period_type : %s, n_periods : %d)" % (self.n_users, self.n_items, self.period_type, self.n_periods))

    def _load_ratings(self, ratingfile, sep = ','):
        ratings = pd.read_csv(ratingfile, sep=sep, header=0)
        # ratings.columns = ['UserId', 'ItemId', 'rating', 'timestamp', 'ctr']
        return ratings
    
    def years2bins(self, itemage):
        bins = [-1] + [0, 2, 4, 7, 10, 14, 20] 
        print("---------- Using Bins: ", bins[1:], "----------")
        itemage_ = np.copy(itemage)
        replaces = np.arange(len(bins) - 1)
        for bidx in range(1, len(bins)):
            b = list(range(bins[bidx-1]+1, bins[bidx]+1))
            itemage_[itemage.isin(b)] = replaces[bidx - 1]
        n_bins = min(self.n_periods, len(replaces))
        return itemage_, n_bins

    def _get_item_birthdate(self):
        d = pd.concat([self.train_full, self.test], ignore_index=True)
        # self.n_items = max(d['ItemId'])+1
        item_birthdate = np.zeros(self.n_items, dtype='int64')
        for i, group in d.groupby(['ItemId']):
            item_birthdate[i] = min(group['timestamp'].to_list())
        # df = pd.DataFrame(data=np.stack((np.arange(self.n_items), item_birthdate), 1), columns = ['ItemId', 'birthdate'])
        # df.to_csv(self.path + '/simulation/item_birthdate.csv', sep=',', header=True, index=False)
        # exit(0)
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
            itemage = ((timestamp - item_pt) * 1.0 / period_second).long().clip(0, max_period)
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
        
        # # use bins to map the itemage
        if self.task == 'OPPT':
            print('For mapping itemage to bins, max_period:', max_period)
            if self.dataset == 'ml-latest-small':
                itemage_ = np.copy(itemage)
                bins = [-1] + [0, 2, 4, 7, 10, 14, 20]
                print("---------- Using Bins: ", bins[1:], "----------")
                replaces = np.arange(len(bins) - 1)
                for bidx in range(1, len(bins)):
                    b = list(range(bins[bidx-1]+1, bins[bidx]+1))
                    itemage_[itemage.isin(b)] = replaces[bidx - 1]
                self.n_bins = min(self.n_periods, len(replaces))
                # print(np.unique(itemage, return_counts=True))
                # print(np.unique(itemage_, return_counts=True), '\n')
                # exit(0)
                return itemage_
        return itemage

    def estimated_predOP_replace(self, mode='b3', static=False):
        df = pd.concat([self.train, self.valid], ignore_index=True)
        scores_train, scores_valid = np.zeros(len(self.train), dtype=float), np.zeros(len(self.valid), dtype=float)
        if mode == 'b3': # popularity at item-age t
            assert static == False
            for T in range(self.n_periods):
                s_T = (df['itemage'] == T).mean()
                scores_train[self.train['itemage'].values == T] = s_T
                scores_valid[self.valid['itemage'].values == T] = s_T
            # print("MSE of %s: %.4f" % (mode, ((self.train['predOP'] - scores_train) ** 2).mean()))
            self.train['predOP'], self.valid['predOP'] = scores_train, scores_valid
            print("!!!!! Estimate and Replace the propensities with B3. Done !!!!!")
        elif mode == 'b2_i': # P(O|i)
            assert static == False
            for i in range(self.n_items):
                s_i = (df['ItemId'] == i).mean()
                scores_train[(self.train['ItemId'].values == i)] = s_i
                scores_valid[(self.valid['ItemId'].values == i)] = s_i
            self.train['predOP'], self.valid['predOP'] = scores_train, scores_valid
            print("!!!!! Estimate and Replace the propensities with pop_i. Done !!!!!")
        elif mode == 'b4': # popularity of item i at item-age t
            assert static == False
            for T in range(self.n_periods):
                for i in range(self.n_items):
                    s_iT = ((df['itemage'] == T) & (df['ItemId'] == i)).mean()
                    scores_train[(self.train['itemage'].values == T) & (self.train['ItemId'].values == i)] = s_iT
                    scores_valid[(self.valid['itemage'].values == T) & (self.valid['ItemId'].values == i)] = s_iT
            # norm = 1.0 / scores_train.mean() * self.train['predOP'].values.mean()
            # scores_train *= norm
            # scores_valid *= norm
            # print((self.train['predOP'].values ** 2).mean(), ((scores_train) ** 2).mean())
            # print("MSE of %s: %.4f" % (mode, ((self.train['predOP'] - scores_train) ** 2).mean()))
            self.train['predOP'], self.valid['predOP'] = scores_train, scores_valid
            print("!!!!! Estimate and Replace the propensities with B4 (pop_it). Done !!!!!")
        elif (mode == 'mf') or (mode == 'tmtf'):
            path = self.path + "/sim-estimOP/predOP_%s_small_0.1.csv"
            predOP = pd.read_csv(path % mode, sep=',', header=0)
            if mode == 'mf':
                predOP_numpy = np.zeros((self.n_users, self.n_items), dtype=float)
                predOP_numpy[predOP['UserId'].values, predOP['ItemId'].values] = predOP['predOP'].values
                self.train['predOP'] = predOP_numpy[self.train['UserId'].values, self.train['ItemId'].values]
                self.valid['predOP'] = predOP_numpy[self.valid['UserId'].values, self.valid['ItemId'].values]
                # print(self.train[self.train['predOP'] == 0.0][:20])
                # exit(0)
            else:
                num_train = len(self.train)
                df = df.sort_values(by=['UserId', 'ItemId', 'itemage'])
                predOP = predOP.sort_values(by=['UserId', 'ItemId', 'itemage'])
                assert ((df['UserId'].values != predOP['UserId'].values).sum() == 0)
                assert ((df['ItemId'].values != predOP['ItemId'].values).sum() == 0)
                # print(df[df['ItemId'].values != predOP['ItemId'].values])
                # print(predOP[df['ItemId'].values != predOP['ItemId'].values])
                assert ((df['itemage'].values != predOP['itemage'].values).sum() == 0)
                df['predOP'] = predOP['predOP'].values
                # print(df[:10], predOP[:10])
                df = df.sort_index()
                train_ = df[:num_train]
                assert (train_['UserId'] != self.train['UserId']).sum() == 0
                assert (train_['ItemId'] != self.train['ItemId']).sum() == 0
                assert (train_['itemage'] != self.train['itemage']).sum() == 0
                self.train['predOP'] = train_['predOP'].values
                self.valid['predOP'] = df[num_train:]['predOP'].values
                # print(self.train[:10])
                # print(train_[:10])
            print("!!!!! Estimate and Replace the propensities with %s. Done !!!!!" % mode)
            # exit(0)
        else:
            print("Unknown estimated predOP")
            exit(1)
            

    def merge_predOP(self, train):
        if self.dataset == 'ml-latest-small':
            # filename = self.path + self.dataset + '/predOP_tmtf_small_0.1.csv'
            filename = self.path + self.dataset + '/predOP_tmtf.csv'
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

    def resplitting_random(self):
        # if os.path.exists(self.path + self.dataset + '/train_random_OPPT_old.csv'):
        #     self.train_full = self._load_ratings(self.path + self.dataset + '/train_random_OPPT_old.csv')
        #     self.test = self._load_ratings(self.path + self.dataset + '/test_random_OPPT_old.csv')
        #     self.train, self.valid = self.train_full, self.train_full
        #     print("Loading randomly-splitted train, valid and test set for OPPT (Task 2). Done.")
        #     return
        if os.path.exists(self.path + self.dataset + '/valid_random_OPPT.csv'):
            self.train_full = None
            self.train = self._load_ratings(self.path + self.dataset + '/train_random_OPPT.csv')
            self.valid = self._load_ratings(self.path + self.dataset + '/valid_random_OPPT.csv')
            self.test = self._load_ratings(self.path + self.dataset + '/test_random_OPPT.csv')
            self.train_full = pd.concat([self.train, self.valid], axis=0, ignore_index=True)
            print("Nr in train, valid, test are %d, %d, %d" % (len(self.train), len(self.valid), len(self.test)))
            n_valid, n_test = len(self.valid), len(self.test)
            self.valid = self.valid.drop(self.valid[~self.valid['ItemId'].isin(self.train['ItemId'].unique())].index)
            self.test = self.test.drop(self.test[~self.test['ItemId'].isin(self.train['ItemId'].unique())].index)
            print("Nr in valid and test whose items are not in trainset: %d and %d" % (n_valid - len(self.valid), n_test - len(self.test)))
            print("Loading randomly-splitted train, valid and test set for OPPT (Task 2). Done.")
            return
        print("Generate the random-splitted training/test set.")
        # # combine train_full and test as a whole dataset
        df = pd.concat([self.train_full, self.test], ignore_index=True)
        # # random splitting by user
        df_train, df_valid, df_test = [], [], []
        np.random.seed(2021)
        for (u, group) in df.groupby(['UserId']):
            group_ = group.sample(frac=1, random_state=np.random.randint(10000))
            n_test = int(0.2 * len(group_))
            n_train = int(0.7 * len(group_))
            df_train.append(group_[:n_train])
            df_valid.append(group_[n_train:-n_test])
            df_test.append(group_[-n_test:])
        self.train_full = None
        self.train = pd.concat(df_train, axis=0)
        self.valid = pd.concat(df_valid, axis=0)
        self.test = pd.concat(df_test, axis=0)
        assert (len(self.train) + len(self.valid) + len(self.test)) == len(df)

        # # # random splitting over all
        # df_ = df.sample(frac=1, random_state=2021)
        # n_train, n_test = int(len(df_) * 0.7), int(len(df_) * 0.2)
        # self.train = df_[:n_train]
        # self.valid = df_[n_train:-n_test]
        # self.test = df_[-n_test:]

        # only keep ['UserId', 'ItemId', 'rating', 'timestamp']
        cols = ['UserId', 'ItemId', 'rating', 'timestamp']
        self.train = self.train[cols]
        self.valid = self.valid[cols]
        self.test = self.test[cols]
        self.train_full = pd.concat([self.train, self.valid], axis=0, ignore_index=True)
        print("Nr in train, valid, test are %d, %d, %d" % (len(self.train), len(self.valid), len(self.test)))
        print("Avg-rating in train, valid, test are %.6f, %.6f, %.6f" % (self.train['rating'].mean(), self.valid['rating'].mean(), self.test['rating'].mean()))
        # self.train.to_csv(self.path + self.dataset + '/train_random_OPPT.csv', index=False)
        # self.valid.to_csv(self.path + self.dataset + '/valid_random_OPPT.csv', index=False)
        # self.test.to_csv(self.path + self.dataset + '/test_random_OPPT.csv', index=False)
        print("For OPPT (task 2), Saving randomly-splitted dataset. Done.")
    
    def interation_data_time_OIPT(self, ratings):
        if self.task.upper() != 'OIPT':
            raise NotImplementedError("Make sure task as OIPT when calling resplitting_time_OIPT")
        # load as interactions
        if os.path.exists(self.path + self.dataset + '/valid_time_OIPT.csv'):
            train = self._load_ratings(self.path + self.dataset + '/train_time_OIPT.csv')
            self.train_interactions = self._df2interactions(train)
            valid = self._load_ratings(self.path + self.dataset + '/valid_time_OIPT.csv')
            self.valid_interactions = self._df2interactions(valid)
            print("Columns of data are, ", train.columns)
            print("Loading time-based-splitted training and valid set. Done.")

            # print("The distributions of p_T in training set:", '\n', '[', ', '.join([str(train['target'][train['itemage'] == t].mean()) for t in range(self.n_periods)]), ']\n')
            # print("The distributions of p_T in valid set:", '\n', '[', ', '.join([str(valid['target'][valid['itemage'] == t].mean()) for t in range(self.n_periods)]), ']\n')
            # exit(0)
            return
        # 
        ob_uiT = np.zeros((self.n_users, self.n_items, self.n_periods), dtype=int)
        # # del item without birthdate, means not be interacted
        # ob_uiT[:, self.item_birthdate == 0] = -1
        for u, group in ratings.groupby(by=['UserId']):
            firsttime = min(group['timestamp'])
            lasttime = max(group['timestamp'])
            smallestT_perI = self.get_itemage(np.arange(self.n_items), np.full(self.n_items, firsttime)) # inside, clip(0, .)
            biggestT_perI = self.get_itemage(np.arange(self.n_items), np.full(self.n_items, lasttime), del_young_items=True) # if item enters system after lasttime, < 0
            # # del items enters the system after lasttime, not available
            ob_uiT[u][self.item_birthdate >= lasttime] = -1
            idx_i, idx_T = [], []
            for i in np.arange(self.n_items)[self.item_birthdate < lasttime]:
                # before user enters system, not available
                idx_T.append(np.arange(smallestT_perI[i]))
                idx_i.append(np.full(smallestT_perI[i], i))
                # after user leaves system, not available
                idx_T.append(np.arange(biggestT_perI[i]+1, self.n_periods))
                idx_i.append(np.full(self.n_periods-(biggestT_perI[i]+1), i))
            if len(idx_i) > 0:
                idx_i = np.concatenate(idx_i)
                idx_T = np.concatenate(idx_T)
                ob_uiT[u][idx_i, idx_T] = -1
        # set pos = 1, (u, i, itemage)
        users, items, itemages = ratings['UserId'], ratings['ItemId'], ratings['ItemAge']
        targets = np.ones_like(users, dtype=int)
        ob_uiT[users, items, itemages] = 1
        # negs
        users_neg, items_neg, itemages_neg = np.where(ob_uiT==0)
        assert users_neg.shape == items_neg.shape
        targets_neg = np.zeros_like(users_neg, dtype=int)
        # merge pos and neg
        users = np.concatenate((users, users_neg), axis=0)
        items = np.concatenate((items, items_neg), axis=0)
        itemages = np.concatenate((itemages, itemages_neg), axis=0)
        targets = np.concatenate((targets, targets_neg), axis=0)
        nums = users.shape[0]
        # numpy to pandas, cols = ['user', 'item', 'itemage', 'target']
        df = pd.DataFrame({'user': users, 'item': items, 'itemage': itemages, 'target': targets})
        self.n_periods = min(self.n_periods, max(itemages) + 1)

        # # shuffle, split and save into file
        # before we use 517 as random seed
        np.random.seed(2021)
        uniform_p = np.random.uniform(size=nums)
        # # take 10% as valid set
        df_train = df[uniform_p <= 0.8]
        df_valid = df[uniform_p > 0.8]
        assert (len(df_train) + len(df_valid)) == len(df)
        print("Nr. in data, train and valid are %d, %d and %d" % (len(df['target']), len(df_train['target']), len(df_valid['target'])))
        print("#o in data, train and valid are %d, %d and %d" % (df['target'].sum(), df_train['target'].sum(), df_valid['target'].sum()))
        print("p(o) in data, train and valid are %.4f, %.4f and %.4f" % (df['target'].mean(), df_train['target'].mean(), df_valid['target'].mean()))
        df_train.to_csv(self.path + self.dataset + '/train_time_OIPT.csv', index=False)
        df_valid.to_csv(self.path + self.dataset + '/valid_time_OIPT.csv', index=False)
        print("Saving time-based-splitted dataset. Done.")
        self.train_interactions, self.test_interactions = self._df2interactions(df_train), self._df2interactions(df_valid)
        

    def resplitting_random_OIPT(self):
        if self.task.upper() != 'OIPT':
            raise NotImplementedError("Make sure task as OIPT when calling resplitting_random_OIPT")

        # load as interactions
        if os.path.exists(self.path + self.dataset + '/valid_random_OIPT.csv'):
            train = self._load_ratings(self.path + self.dataset + '/train_random_OIPT.csv')
            self.train_interactions = self._df2interactions(train)
            valid = self._load_ratings(self.path + self.dataset + '/valid_random_OIPT.csv')
            self.valid_interactions = self._df2interactions(valid)
            test = self._load_ratings(self.path + self.dataset + '/test_random_OIPT.csv')
            self.test_interactions = self._df2interactions(test)
            # cols = ['user', 'item', 'itemage', 'target'] which are generated by negative sampling. Not timestamp, because for negs, we do not have their timestamps
            self.n_users = max(max(train['user']), max(valid['user']), max(test['user'])) + 1
            self.n_items = max(max(train['item']), max(valid['item']), max(test['item'])) + 1
            self.n_periods = max(max(train['itemage']), max(valid['itemage']), max(test['itemage'])) + 1
            self.max_period = self.n_periods - 1
            print("Columns of data are, ", train.columns)
            print("Loading randomly-splitted training and test set. Done.")
            return
        
        # generate negs, merge with pos, shuffle, split, and save
        # need item_birthdate
        ratings = pd.concat([self.train_full, self.test], ignore_index=True)
        self.n_users, self.n_items = max(ratings['UserId']) + 1, max(ratings['ItemId']) + 1
        self.max_period = self.n_periods - 1
        self.item_birthdate = self._get_item_birthdate()
        self.train_interactions, self.valid_interactions, self.test_interactions = self._neg_sampling_time_based(ratings)
        print("Generate the random-splitted training/test set after negative generation. Done.")
        
    def resplitting_random_OIPT3(self, ratio=0.1):
        midname = '4mf' # or ''
        if self.task.upper() != 'OIPT':
            raise NotImplementedError("Make sure task as OIPT when calling resplitting_random_OIPT")
        print("!!!!!!! We only use %f data for train and %f for validation, and the rest for test for better generation for TART !!!!!!!!" % (ratio, ratio))
        if os.path.exists(self.path + self.dataset + '/valid_random_OIPT%s_task3.csv' % midname):
            train = self._load_ratings(self.path + self.dataset + '/train_random_OIPT%s_task3.csv' % midname)
            self.train_interactions = self._df2interactions(train)
            valid = self._load_ratings(self.path + self.dataset + '/valid_random_OIPT%s_task3.csv' % midname)
            self.valid_interactions = self._df2interactions(valid)
            test = self._load_ratings(self.path + self.dataset + '/test_random_OIPT%s_task3.csv' % midname)
            self.test_interactions = self._df2interactions(test)
            # cols = ['user', 'item', 'itemage', 'target'] which are generated by negative sampling. Not timestamp, because for negs, we do not have their timestamps
            self.n_users = max(max(train['user']), max(valid['user']), max(test['user'])) + 1
            self.n_items = max(max(train['item']), max(valid['item']), max(test['item'])) + 1
            self.n_periods = max(max(train['itemage']), max(valid['itemage']), max(test['itemage'])) + 1
            self.max_period = self.n_periods - 1
            print("Columns of data are, ", train.columns)
            print("Loading randomly-splitted training and test set. Done.")
            return 

        # load as interactions
        if os.path.exists(self.path + self.dataset + '/valid_random_OIPT.csv'):
            train = self._load_ratings(self.path + self.dataset + '/train_random_OIPT.csv')
            valid = self._load_ratings(self.path + self.dataset + '/valid_random_OIPT.csv')
            test = self._load_ratings(self.path + self.dataset + '/test_random_OIPT.csv')
            # cols = ['user', 'item', 'itemage', 'target'] which are generated by negative sampling. Not timestamp, because for negs, we do not have their timestamps
            self.n_users = max(max(train['user']), max(valid['user']), max(test['user'])) + 1
            self.n_items = max(max(train['item']), max(valid['item']), max(test['item'])) + 1
            self.n_periods = max(max(train['itemage']), max(valid['itemage']), max(test['itemage'])) + 1
            self.max_period = self.n_periods - 1
            df = pd.concat([train, valid, test], axis=0)
            assert len(df) == (len(train) + len(valid) + len(test))
            np.random.seed(517)
            indice = np.random.uniform(size=(len(df)))
            indice_train = (indice <= 0.1)
            indice_valid = (indice > 0.9)
            indice_test = ((indice > 0.1) & (indice <= 0.9))
            self.train_interactions = self._df2interactions(df[indice_train])
            self.valid_interactions = self._df2interactions(df[indice_valid])
            self.test_interactions = self._df2interactions(df[indice_test])
            df[indice_train].to_csv(self.path + self.dataset + '/train_random_OIPT_task3.csv', index=False)
            df[indice_valid].to_csv(self.path + self.dataset + '/valid_random_OIPT_task3.csv', index=False)
            df[indice_test].to_csv(self.path + self.dataset + '/test_random_OIPT_task3.csv', index=False)
        else:
            raise "Errors"
        
        
    def _neg_sampling_time_based(self, ratings):
        ob_uiT = np.zeros((self.n_users, self.n_items, self.n_periods), dtype=int)
        # # del item without birthdate, means not be interacted
        ob_uiT[:, self.item_birthdate == 0] = -1
        for u, group in ratings.groupby(by=['UserId']):
            firsttime = min(group['timestamp'])
            lasttime = max(group['timestamp'])
            smallestT_perI = self.get_itemage(np.arange(self.n_items), np.full(self.n_items, firsttime)) # inside, clip(0, .)
            biggestT_perI = self.get_itemage(np.arange(self.n_items), np.full(self.n_items, lasttime), del_young_items=True) # if item enters system after lasttime, < 0
            # # del items enters the system after lasttime, not available
            ob_uiT[u][self.item_birthdate >= lasttime] = -1
            idx_i, idx_T = [], []
            for i in np.arange(self.n_items)[self.item_birthdate < lasttime]:
                # before user enters system, not available
                idx_T.append(np.arange(smallestT_perI[i]))
                idx_i.append(np.full(smallestT_perI[i], i))
                # after user leaves system, not available
                idx_T.append(np.arange(biggestT_perI[i]+1, self.n_periods))
                idx_i.append(np.full(self.n_periods-(biggestT_perI[i]+1), i))
            if len(idx_i) > 0:
                idx_i = np.concatenate(idx_i)
                idx_T = np.concatenate(idx_T)
                ob_uiT[u][idx_i, idx_T] = -1
        # set pos = 1, (u, i, itemage)
        users, items = ratings['UserId'], ratings['ItemId']
        itemages = self.get_itemage(ratings['ItemId'], ratings['timestamp'])
        targets = np.ones_like(users, dtype=int)
        ob_uiT[users, items, itemages] = 1

        # negs
        users_neg, items_neg, itemages_neg = np.where(ob_uiT==0)
        assert users_neg.shape == items_neg.shape
        targets_neg = np.zeros_like(users_neg, dtype=int)

        # merge pos and neg
        users = np.concatenate((users, users_neg), axis=0)
        items = np.concatenate((items, items_neg), axis=0)
        itemages = np.concatenate((itemages, itemages_neg), axis=0)
        targets = np.concatenate((targets, targets_neg), axis=0)
        nums = users.shape[0]
        # numpy to pandas, cols = ['user', 'item', 'itemage', 'target']
        df = pd.DataFrame({'user': users, 'item': items, 'itemage': itemages, 'target': targets})
        self.n_periods = min(self.n_periods, max(itemages) + 1)

        # shuffle, split and save into file
        np.random.seed(2021)
        uniform_p = np.random.uniform(size=nums)
        # ratio = [0.7, 0.8, 1.0] # following 7:1:2 for train/valid/test splitting
        df_train = df[uniform_p <= 0.7]
        df_valid = df[(uniform_p > 0.7) & (uniform_p <= 0.8)]
        df_test = df[uniform_p > 0.8]
        assert (len(df_train) + len(df_valid) + len(df_test)) == len(df)
        print("Nr. in data, train, valid and test are %d, %d, %d and %d" % (len(df['target']), len(df_train['target']), len(df_valid['target']), len(df_test['target'])))
        print("#o in data, train, valid and test are %d, %d, %d and %d" % (df['target'].sum(), df_train['target'].sum(), df_valid['target'].sum(), df_test['target'].sum()))
        print("p(o) in data, train, valid and test are %.4f, %.4f, %.4f and %.4f" % (df['target'].mean(), df_train['target'].mean(), df_valid['target'].mean(), df_test['target'].mean()))
        df_train.to_csv(self.path + self.dataset + '/train_random_OIPT.csv', index=False)
        df_valid.to_csv(self.path + self.dataset + '/valid_random_OIPT.csv', index=False)
        df_test.to_csv(self.path + self.dataset + '/test_random_OIPT.csv', index=False)
        print("Saving randomly-splitted dataset. Done.")
        return self._df2interactions(df_train), self._df2interactions(df_valid), self._df2interactions(df_test)

    def _df2interactions(self, df):
        interaction = {}
        interaction['user'] = df['user'].values
        interaction['item'] = df['item'].values
        interaction['itemage'] = df['itemage'].values
        interaction['target'] = df['target'].values
        interaction['num'] = len(df)
        return interaction
