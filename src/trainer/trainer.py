import os
import itertools
import torch
import torch.optim as optim
import numpy as np
import math
import matplotlib.pyplot as plt
from time import time

import sys
sys.path.append('../src/')
from evaluator.evaluator import calculate_metrics, cal_ratpred_metrics, cal_op_metrics

class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model, data):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.data = data
        # self.train = self.data.train_full
        self.learner = config['optimizer']
        self.learning_rate = float(config['learning_rate'])
        self.l2_reg = 0
        if 'l2_reg' in config:
            self.l2_reg = float(config['l2_reg'])

        self.best_valid_score = None
    
    def _build_optimizer(self):
        r"""Init the Optimizer
        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)
        return optimizer

    def _early_stopping(self, value, best, cur_step, max_step, bigger=True):
        r""" validation-based early stopping
        Args:
            value (float): current result
            best (float): best result
            cur_step (int): the number of consecutive steps that did not exceed the best result
            max_step (int): threshold steps for stopping
            bigger (bool, optional): whether the bigger the better
        Returns:
            tuple:
            - float, best result after this step
            - int, the number of consecutive steps that did not exceed the best result after this step
            - bool, whether to stop
            - bool, whether to update
        """
        stop_flag = False
        update_flag = False
        if best is None:
            cur_step = 0
            best = value
            update_flag = True
            return best, cur_step, stop_flag, update_flag
        if bigger:
            if value > best:
                cur_step = 0
                best = value
                update_flag = True
            else:
                cur_step += 1
                if cur_step > max_step:
                    stop_flag = True
        else:
            if value < best:
                cur_step = 0
                best = value
                update_flag = True
            else:
                cur_step += 1
                if cur_step > max_step:
                    stop_flag = True
        return best, cur_step, stop_flag, update_flag

    def fit(self):
        r"""Train the model based on the train data.
        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.
        """

        raise NotImplementedError('Method [next] should be implemented.')
    def save_model(self, epoch):
        state = {'net': self.model.state_dict(), 'optimizer':self.optimizer.state_dict(), 'epoch':epoch}
        torch.save(state, self.saved_model_file)
    
    def load_model(self):
        checkpoint = torch.load(self.saved_model_file)
        self.model.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


class TARS_Trainer(AbstractTrainer):
    '''
    For time-aware recommender, such as TMF
    '''
    def __init__(self, config, model, data):
        super(TARS_Trainer, self).__init__(config, model, data)

        self.optimizer = config['optimizer']
        if config['debiasing']:
            self.debiasing = "_ips"
        else:
            self.debiasing = '_naive'
        self.saved_model_file = "./checkpoint_dir/" + config['dataset'] + self.debiasing # config['checkpoint_dir']
        print("model will be saved into:", self.saved_model_file)
        self.epochs = config['epochs']
        self.batch_size = int(config['batch_size'])
        
        self.optimizer = self._build_optimizer()

    def _data_pre(self, train_set):
        train = train_set
        uid_list = torch.from_numpy(np.array(train['UserId'].values, dtype=int)).to(self.device)
        iid_list = torch.from_numpy(np.array(train['ItemId'].values, dtype=int)).to(self.device)
        target = torch.from_numpy(np.array(train['rating'].values, dtype=float)).to(self.device)
        itemage = torch.from_numpy(np.array(train['ItemAge'].values, dtype=int)).to(self.device)
        ctr = torch.from_numpy(np.array(train['ctr'].values, dtype=float)).to(self.device)
        
        interaction = {}
        interaction['user'] = uid_list
        interaction['item'] = iid_list
        interaction['target'] = target
        interaction['ctr'] = ctr
        interaction['itemage'] = itemage

        interaction['num'] = uid_list.size()[0]
        return interaction

    def _train_epoch(self, interaction, shuffle_data=True):
        if shuffle_data:
            # shuffle the data
            order = torch.randperm(interaction['num'])
            for key in ['user', 'item', 'itemage', 'target', 'ctr']:
                value = interaction[key]
                interaction[key] = value[order]
        
        user = interaction['user']
        item = interaction['item']
        target = interaction['target']
        ctr = interaction['ctr']
        itemage = interaction['itemage']
        
        # batch split
        num_batch = math.ceil(interaction['num'] * 1.0 / self.batch_size)
        start_idxs, end_idxs = [], []
        for i_batch in range(num_batch - 1):
            start_idxs.append(i_batch * self.batch_size)
            end_idxs.append((i_batch+1) * self.batch_size)
        start_idxs.append((num_batch - 1) * self.batch_size)
        end_idxs.append(interaction['num'])
        start_idxs = np.array(start_idxs, dtype=int)
        end_idxs = np.array(end_idxs, dtype=int)

        # train on batch
        total_loss = 0
        for i_batch in range(len(start_idxs)):
            start_idx, end_idx = start_idxs[i_batch], end_idxs[i_batch]
            self.optimizer.zero_grad()
            losses = self.model.calculate_loss({'user': user[start_idx:end_idx], \
                'item': item[start_idx:end_idx], 'target': target[start_idx:end_idx], \
                'ctr': ctr[start_idx:end_idx], 'itemage': itemage[start_idx:end_idx]})
            total_loss += losses.item()
            losses.backward()
            self.optimizer.step()
        return total_loss

    def fit(self, valid_data=None, verbose=True, saved=True):
        interaction = self._data_pre(self.train)
        start = time()
        for epoch_idx in range(self.epochs):
            train_loss = self._train_epoch(interaction)
            if (epoch_idx + 1) % 1 == 0: # evaluate on valid set
                # self.load_model()
                valid_results, valid_loss = self.evaluate()
                print("epoch %d, time-consumin: %f s, train-loss: %f, valid-loss: %f, \nresults on validset: %s" % (epoch_idx+1, time()-start, train_loss, valid_loss, str(valid_results)), flush=True)
                self.best_valid_score, _, stop_flag, _ = self._early_stopping(valid_loss, self.best_valid_score, epoch_idx, 10, bigger=False)
                # print(self.best_valid_score, stop_flag, valid_loss)
                # exit(0)
                if stop_flag:
                    print("Finished training, best eval result in epoch %d" % epoch_idx)
                    break
                start = time()
        self.save_model(epoch_idx)
        return self.model

    @torch.no_grad()      
    def _eval_epoch(self, interaction):
        user = interaction['user']
        item = interaction['item']
        target = interaction['target']
        ctr = interaction['ctr']
        itemage = interaction['itemage']
        
        # batch split
        num_batch = math.ceil(interaction['num'] * 1.0 / self.batch_size)
        start_idxs, end_idxs = [], []
        for i_batch in range(num_batch - 1):
            start_idxs.append(i_batch * self.batch_size)
            end_idxs.append((i_batch+1) * self.batch_size)
        start_idxs.append((num_batch - 1) * self.batch_size)
        end_idxs.append(interaction['num'])
        start_idxs = np.array(start_idxs, dtype=int)
        end_idxs = np.array(end_idxs, dtype=int)

        # eval on batch
        total_loss = 0

        for i_batch in range(len(start_idxs)):
            start_idx, end_idx = start_idxs[i_batch], end_idxs[i_batch]
            self.optimizer.zero_grad()
            losses = self.model.calculate_loss({'user': user[start_idx:end_idx], \
                'item': item[start_idx:end_idx], 'itemage':itemage[start_idx:end_idx]})
            total_loss += losses.item()
            losses.backward()
            self.optimizer.step()
        return total_loss
    
    @torch.no_grad()
    def evaluate(self):
        interaction = self._data_pre(self.data.valid)
        # losses
        losses = self.model.calculate_loss(interaction).item()
        # results
        scores = self.model.predict(interaction)
        # find the position of the target item
        targets = interaction['target']
        results = cal_ratpred_metrics(scores.cpu().numpy(), targets.cpu().numpy())
        return results, losses
        

class Seq_Trainer(AbstractTrainer):
    def __init__(self, config, model, data):
        super(Seq_Trainer, self).__init__(config, model, data)

        self.optimizer = config['optimizer']
        if config['debiasing']:
            self.debiasing = "_ips"
        else:
            self.debiasing = '_naive'
        self.saved_model_file = "./checkpoint_dir/" + config['dataset'] + '_' + config['mode'] + self.debiasing # config['checkpoint_dir']
        print("model will be saved into:", self.saved_model_file)
        self.epochs = config['epochs']
        self.batch_size = int(config['batch_size'])
        self.max_item_list_len = int(config['max_item_list_len'])

        self.optimizer = self._build_optimizer()


    def _data_pre(self, train_set):
        train = train_set
        n_items = self.data.n_items
        uid_list, seq_list, seq_len, ctr_target, target = [], [], [], [], []
        for u, group in train.groupby(['UserId']):
            u_ratings = group.sort_values(['timestamp'], ascending=True)
            u_items = u_ratings['ItemId'].values
            u_ctrs = u_ratings['ctr'].values
            for idx in range(len(u_items)-1):
                uid_list.append(u)
                ctr_target.append(u_ctrs[idx+1])
                target.append(u_items[idx+1])
                if idx >= self.max_item_list_len:
                    seq_list.append(u_items[(idx-self.max_item_list_len + 1):(idx + 1)])
                    seq_len.append(self.max_item_list_len)
                else:
                    padding_list = [n_items] * (self.max_item_list_len - idx - 1)
                    # print(u_items[:(idx+1)], padding_list)
                    seq_list.append(list(u_items[:(idx+1)]) + padding_list)
                    seq_len.append(idx+1)
                # print(seq_list, len(seq_list[-1]))
                # assert len(seq_list[-1]) == self.max_item_list_len
        uid_list = torch.from_numpy(np.array(uid_list, dtype=int)).to(self.device)
        seq_list = torch.from_numpy(np.array(seq_list, dtype=int)).to(self.device)
        seq_len = torch.from_numpy(np.array(seq_len, dtype=int)).to(self.device)
        target = torch.from_numpy(np.array(target, dtype=int)).to(self.device)
        ctr_target = torch.from_numpy(np.array(ctr_target, dtype=float)).to(self.device)
        interaction = {}
        interaction['seq'] = seq_list
        interaction['seq_len'] = seq_len
        interaction['target'] = target
        interaction['ctr'] = ctr_target
        assert seq_list.size()[0] == ctr_target.size()[0]
        interaction['num'] = seq_list.size()[0]
        return interaction

    def _data_pre_fullseq(self, train_full):
        train = train_full
        n_items = self.data.n_items
        uid_list, seq_list, seq_len, ctr_target, target = [], [], [], [], []
        for u, group in train.groupby(['UserId']):
            u_ratings = group.sort_values(['timestamp'], ascending=True)
            u_items = u_ratings['ItemId'].values
            u_ctrs = u_ratings['ctr'].values
            u_n_items = len(u_items) - 1
            target.append(u_items[-1])
            ctr_target.append(u_ctrs[-1])
            u_items = u_items[:-1]
            uid_list.append(u)
            if u_n_items >= self.max_item_list_len:
                seq_list.append(u_items[-self.max_item_list_len:])
                seq_len.append(self.max_item_list_len)
            else:
                padding_list = [n_items] * (self.max_item_list_len - u_n_items)
                seq_list.append(list(u_items) + padding_list)
                seq_len.append(u_n_items)

        uid_list = torch.from_numpy(np.array(uid_list, dtype=int)).to(self.device)
        seq_list = torch.from_numpy(np.array(seq_list, dtype=int)).to(self.device)
        seq_len = torch.from_numpy(np.array(seq_len, dtype=int)).to(self.device)
        target = torch.from_numpy(np.array(target, dtype=int)).to(self.device)
        ctr_target = torch.from_numpy(np.array(ctr_target, dtype=float)).to(self.device)
        interaction = {}
        interaction['seq'] = seq_list
        interaction['seq_len'] = seq_len
        interaction['target'] = target
        interaction['ctr'] = ctr_target
        assert seq_list.size()[0] == seq_len.size()[0]
        interaction['num'] = seq_list.size()[0]
        return interaction

    def _train_epoch(self, interaction, shuffle_data=True):
        # print(interaction['num'])
        # exit(0)
        if shuffle_data:
            # shuffle the data
            order = torch.randperm(interaction['num'])
            for key in ['seq', 'seq_len', 'target', 'ctr']:
                value = interaction[key]
                interaction[key] = value[order]
        
        seq_list = interaction['seq']
        seq_len = interaction['seq_len']
        target = interaction['target']
        ctr = interaction['ctr']
        
        # batch split
        num_batch = math.ceil(interaction['num'] * 1.0 / self.batch_size)
        start_idxs, end_idxs = [], []
        for i_batch in range(num_batch - 1):
            start_idxs.append(i_batch * self.batch_size)
            end_idxs.append((i_batch+1) * self.batch_size)
        start_idxs.append((num_batch - 1) * self.batch_size)
        end_idxs.append(interaction['num'])
        start_idxs = np.array(start_idxs, dtype=int)
        end_idxs = np.array(end_idxs, dtype=int)

        # train on batch
        total_loss = 0
        for i_batch in range(len(start_idxs)):
            start_idx, end_idx = start_idxs[i_batch], end_idxs[i_batch]
            self.optimizer.zero_grad()
            losses = self.model.calculate_loss({'seq': seq_list[start_idx:end_idx], \
                'seq_len': seq_len[start_idx:end_idx], 'target': target[start_idx:end_idx], \
                'ctr': ctr[start_idx:end_idx]})
            total_loss += losses.item()
            losses.backward()
            self.optimizer.step()
        return total_loss


    def fit(self, valid_data=None, verbose=True, saved=True):
        interaction = self._data_pre(self.train)
        start = time()
        for epoch_idx in range(self.epochs):
            train_loss = self._train_epoch(interaction)
            if (epoch_idx + 1) % 1 == 0: # evaluate on valid set
                # self.load_model()
                valid_results, valid_loss = self.evaluate()
                print("epoch %d, time-consumin: %f s, train-loss: %f, valid-loss: %f, \nresults on validset: %s" % (epoch_idx+1, time()-start, train_loss, valid_loss, str(valid_results)))
                self.best_valid_score, _, stop_flag, _ = self._early_stopping(valid_loss, self.best_valid_score, epoch_idx, 10, bigger=False)
                # print(self.best_valid_score, stop_flag, valid_loss)
                # exit(0)
                if stop_flag:
                    print("Finished training, best eval result in epoch %d" % epoch_idx)
                    break
                self.save_model(epoch_idx)
                start = time()
        return self.model

    @torch.no_grad()      
    def _eval_epoch(self, interaction):
        seq_list = interaction['seq']
        seq_len = interaction['seq_len']
        target = interaction['target']
        ctr = interaction['ctr']
        
        # batch split
        num_batch = math.ceil(interaction['num'] * 1.0 / self.batch_size)
        start_idxs, end_idxs = [], []
        for i_batch in range(num_batch - 1):
            start_idxs.append(i_batch * self.batch_size)
            end_idxs.append((i_batch+1) * self.batch_size)
        start_idxs.append((num_batch - 1) * self.batch_size)
        end_idxs.append(interaction['num'])
        start_idxs = np.array(start_idxs, dtype=int)
        end_idxs = np.array(end_idxs, dtype=int)

        # eval on batch
        total_loss = 0

        for i_batch in range(len(start_idxs)):
            start_idx, end_idx = start_idxs[i_batch], end_idxs[i_batch]
            scores = self.self.model.full_sort_predict({'seq': seq_list[start_idx:end_idx], \
                'seq_len': seq_len[start_idx:end_idx]})
            self.optimizer.zero_grad()
            losses = self.model.calculate_loss({'seq': seq_list[start_idx:end_idx], \
                'seq_len': seq_len[start_idx:end_idx], 'target': target[start_idx:end_idx], \
                'ctr': ctr[start_idx:end_idx]})
            total_loss += losses.item()
            losses.backward()
            self.optimizer.step()
        return total_loss

    def evaluate(self):
        interaction = self._data_pre_fullseq(self.data.train_full)
        # losses
        losses = self.model.calculate_loss(interaction).item()
        # results
        scores = self.model.full_sort_predict(interaction)
        # find the position of the target item
        targets = interaction['target']
        target_scores = torch.gather(scores, 1, targets.view(-1, 1)) # [B 1]
        target_pos = (scores >= target_scores).sum(-1) # [B]
        results = calculate_metrics(target_pos.cpu().numpy())
        return results, losses
        
class OP_Trainer(AbstractTrainer):
    '''
    For time-aware observation prediction, such as TMF
    '''
    def __init__(self, config, model, data):
        super(OP_Trainer, self).__init__(config, model, data)

        self.splitting = config['splitting']
        self.optimizer = config['optimizer']
        self.saved_model_file = "./checkpoint_dir/" + config['dataset'] + '_' + config['mode'] + '_ObsPred'
        print("model will be saved into:", self.saved_model_file)
        self.epochs = config['epochs']
        self.batch_size = int(config['batch_size'])
        self.n_users = self.data.n_users
        self.n_items = self.data.n_items
        self.n_periods = self.data.n_periods

        self.optimizer = self._build_optimizer()
        if self.splitting == 'random':
            return

        self.item_birthdate = torch.from_numpy(self.data._get_item_birthdate()).to(self.device)
        # if config['mode'].lower() in ['tmf','tf']:
        #     self.ns_type = 'random'
        # else:
        #     self.ns_type = 'random'
        if 'ns_type' not in config:
            self.ns_type = 'random'
        else:
            self.ns_type = config['ns_type'].lower()
        print("Training with Negative-Sampling: %s based" % self.ns_type)
        if 'time' in self.ns_type:
            self.time_offset = 1
            print("*********** Training with time-based negative sampling (Assumption: after %d month the user lose attention)" % self.time_offset)

        self.train_user_pos = {}
        for u, group in self.data.train_full.groupby(by=['UserId']):
            self.train_user_pos[u] = group['ItemId'].values


    def _data_pre(self, train_set):
        train = train_set
        uid_list = torch.from_numpy(np.array(train['UserId'].values, dtype=int)).to(self.device)
        iid_list = torch.from_numpy(np.array(train['ItemId'].values, dtype=int)).to(self.device)
        target = torch.ones_like(iid_list).to(self.device)
        itemage = torch.from_numpy(np.array(train['ItemAge'].values, dtype=int)).to(self.device)
        timestamp = torch.from_numpy(np.array(train['timestamp'].values, dtype=int)).to(self.device)

        interaction = {}
        interaction['user'] = uid_list
        interaction['item'] = iid_list
        interaction['target'] = target
        interaction['itemage'] = itemage
        interaction['timestamp'] = timestamp
        interaction['num'] = uid_list.size()[0]
        return interaction


    def _merge_interactions(self, inte1, inte2):
        inte_ = {}
        keys = list(set(inte1.keys()) & set(inte2.keys()))
        for key in keys:
            if key == 'num':
                inte_[key] = inte1[key] + inte2[key]
            else:
                try:
                    inte_[key] = torch.cat((inte1[key], inte2[key]), 0)
                except:
                    print(key)
                    exit(1)
        return inte_


    def _neg_sampling_random(self, train_set, full_negs = False):
        interaction_neg = {}
        users, items, itemages = [], [], []
        for u, group in train_set.groupby(by=['UserId']):
            pos_items = group['ItemId'].values
            neg_prob = np.random.uniform(0, 1, size=(self.n_items,))
            neg_prob[pos_items] = -1.
            if full_negs:
                neg_items = np.arange(self.n_items)[neg_prob >= 0.0]
            else:
                neg_items = np.argsort(neg_prob)[-len(pos_items) : ]
            users.append(np.repeat(u, len(neg_items)))
            items.append(neg_items)
            itemages.append(np.random.randint(self.n_periods, size=neg_items.shape)) # randomly generate the itemage
        users = np.concatenate(users, axis=0).astype(int)
        items = np.concatenate(items, axis=0).astype(int)
        itemages = np.concatenate(itemages, axis=0).astype(int)
        targets = np.repeat(0, len(users)).astype(float)
        
        interaction_neg['user'] = torch.from_numpy(users).to(self.device)
        interaction_neg['item'] = torch.from_numpy(items).to(self.device)
        interaction_neg['target'] = torch.from_numpy(targets).to(self.device)
        interaction_neg['itemage'] = torch.from_numpy(itemages).to(self.device)
        interaction_neg['num'] = len(users)
        return interaction_neg

    def _neg_sampling_time(self, interaction, full_negs=False, bigger=True):
        # only add the negatives of different timestamp/itemage, O(T*#Ratings)
        users = interaction['user']
        items = interaction['item']
        num = interaction['num']
        itemages = interaction['itemage']
        
        interaction_neg = {}
        if full_negs:
            ui_ia = torch.zeros((num, self.n_periods), dtype=int)
            ui_ia[range(num), itemages] = 1
            id1, id2 = torch.where(ui_ia == 0)
            users_neg = users[id1]
            items_neg = items[id1]
            itemages_neg = id2.to(self.device)
            targets_neg = torch.zeros_like(id2, dtype=float).to(self.device)
        else:
            # simply add itemage
            users_neg = users
            items_neg = items
            itemages_neg = (itemage + torch.randint(self.time_offset, self.data.n_periods, size=(num, )).to(self.device)).clip(0, self.data.n_periods-1)
            targets_neg = (itemages == itemages_neg).int()

        interaction_neg['num'] = len(users_neg)
        interaction_neg['user'] = users_neg
        interaction_neg['item'] = items_neg
        interaction_neg['target'] = targets_neg
        interaction_neg['itemage'] = itemages_neg
        return interaction_neg

    def _neg_sampling_time_based(self):
        # negatives for training Time-based baselines, O(U*I*T)
        train = self.data.train_full
        ob_uiT = np.zeros((self.n_users, self.n_items, self.n_periods), dtype=int) # O(U*I*T)
        # give all pos (u, i, itemage) label=1
        users, items, itemages = train['UserId'], train['ItemId'], train['ItemAge']
        # # negatives
        # u_lifelength = []
        # n_nnnegs = 0
        for u, group in train.groupby(by=['UserId']):
            lasttime = max(group['timestamp'])
            # # # dropout the items which are published after the lasttime per user
            # idx1 = self.data.item_birthdate >= lasttime 
            # items_neg1 = np.where(idx1)[0]
            # ob_uiT[u][items_neg1] = -1
            # # # for each unobserved (u, i), we have L(u, i, all T before lastT_i) = 1, and L(u, i, after lastT_i) = -1
            # unobserved_items = np.ones(self.n_items)
            # unobserved_items[group['ItemId']] = 0 # delete observed items
            # unobserved_items[idx1] = 0 # delete too young items
            # unobserved_items = np.where(unobserved_items)[0]
            # # get all T before lasttime
            # biggestT_perI = self.data.get_itemage(unobserved_items, np.full(unobserved_items.shape, lasttime))
            # idx_i, idx_T = [], []
            # for i in range(len(unobserved_items)):
            #     if (biggestT_perI[i]+1) >= self.n_periods:
            #         continue
            #     idx_T.append(np.arange(biggestT_perI[i]+1, self.n_periods))
            #     idx_i.append(np.full(self.n_periods-(biggestT_perI[i]+1), unobserved_items[i]))
            # if len(idx_i) > 0:
            #     idx_i = np.concatenate(idx_i)
            #     idx_T = np.concatenate(idx_T)
            #     ob_uiT[u][idx_i, idx_T] = -1
            # # for (u, any i, t before user first) =-1
            firsttime = min(group['timestamp'])
            lasttime = max(group['timestamp'])
            # u_lifelength.append(lasttime-firsttime)

            smallestT_perI = self.data.get_itemage(np.arange(self.n_items), np.full(self.n_items, firsttime)) # inside, clip(0, .)
            biggestT_perI = self.data.get_itemage(np.arange(self.n_items), np.full(self.n_items, lasttime), del_young_items=True) # if item enters system after lasttime, < 0
            # if items enters the system after lasttime, not available
            ob_uiT[u][self.data.item_birthdate >= lasttime] = -1 
            idx_i, idx_T = [], []
            for i in np.arange(self.n_items)[self.data.item_birthdate < lasttime]:
                # Before user enters the system or after he leaves, item with age = [0, itemage in firsttime] & [itemage in lasttime, n_period]= -1
                # n_nnnegs += biggestT_perI[i] - smallestT_perI[i] + 1
                idx_T.append(np.arange(smallestT_perI[i]))
                idx_i.append(np.full(smallestT_perI[i], i))
                # idx_T.append(np.arange(smallestT_perI[i]))
                # idx_i.append(np.full(smallestT_perI[i], i))
                # if item enters the system before lasttime, [itemage in lasttime, n_periods] not available
                idx_T.append(np.arange(biggestT_perI[i]+1, self.n_periods))
                idx_i.append(np.full(self.n_periods-(biggestT_perI[i]+1), i))
                # if biggestT_perI[i] < 0: # if item enters the system after lasttime, all 1-T is not available (-1)
                #     idx_T.append(np.arange(self.n_periods))
                #     idx_i.append(np.full(self.n_periods, i))
                # else: # if item enters the system before lasttime, [itemage in lasttime, n_periods] not available
                #     if ((biggestT_perI[i] + 1) < self.n_periods):
                #         idx_T.append(np.arange(biggestT_perI[i]+1, self.n_periods))
                #         idx_i.append(np.full(self.n_periods-(biggestT_perI[i]+1), i))
            if len(idx_i) > 0:
                idx_i = np.concatenate(idx_i)
                idx_T = np.concatenate(idx_T)
                ob_uiT[u][idx_i, idx_T] = -1
        ob_uiT[users, items, itemages] = 1 # positives set 1
        # u_lifelength_ = (np.array(u_lifelength) / (365 * 24 * 60 * 60)).astype(int)
        # print("user life-length (unit: year): ", np.unique(u_lifelength_, return_counts=True))

        # # ob_uiT: l=0 negatives; l=1 positives, l=-1 meaningless samples.
        users_neg, items_neg, itemages_neg = np.where(ob_uiT==0)
        assert users_neg.shape == items_neg.shape
        users_neg = torch.from_numpy(users_neg).to(self.device)
        items_neg = torch.from_numpy(items_neg).to(self.device)
        itemages_neg = torch.from_numpy(itemages_neg).to(self.device)
        targets_neg = torch.zeros_like(users_neg, dtype=float).to(self.device)

        # # output the distribution of itemage in training set
        print("The distributions of p_T in train set:", '\n', '[', ', '.join([str((itemages.values == t).sum() * 1.0 / ((itemages.values == t).sum() + (itemages_neg == t).cpu().numpy().sum())) for t in range(self.n_periods)]), ']')
        # p_T = []
        # for t in range(self.n_periods):
        #     n_pos, n_neg = (itemages.values == t).sum(), (itemages_neg == t).cpu().numpy().sum()
        #     # print(t, n_pos, n_neg)
        #     p_T.append(n_pos * 1.0 / (n_pos + n_neg))
        # print("The distributions of p_T in training set:", '\n', '[', ', '.join([str(x) for x in p_T]), ']')
        # exit(0)

        interaction_neg = {}
        interaction_neg['num'] = len(users_neg)
        interaction_neg['user'] = users_neg
        interaction_neg['item'] = items_neg
        interaction_neg['target'] = targets_neg
        interaction_neg['itemage'] = itemages_neg
        return interaction_neg

    def _neg_sampling(self, interaction, train_set):
        ns_type = self.ns_type
        # interaction_ = {}
        # users = interaction['user']
        # items = interaction['item']
        # targets = interaction['target']
        # timestamp = interaction['timestamp']
        # itemage = interaction['itemage']

        if ns_type == 'random':
            # # generate data for negs
            # interaction_['user'] = torch.cat((users, users), 0)
            # negs = torch.randint(self.n_items, size=(interaction['num'],)).to(self.device)
            # interaction_['item'] = torch.cat((items, negs), 0)
            # target_neg = (items == negs).int()
            # interaction_['target'] = torch.cat((targets, target_neg), 0)
            # itemage_neg = self.data.get_itemage(items, timestamp, self.item_birthdate)
            # # itemage_neg = ((timestamp - self.item_birthdate[items]) * 1.0 / (30*24*60*60)).int().clip(0, self.data.n_months - 1) # unit: month
            # interaction_['itemage'] = torch.cat((itemage, itemage_neg), 0)
            # interaction_['num'] = interaction['num'] * 2
            interaction_neg = self._neg_sampling_random(train_set, full_negs=True)
            interaction_ = self._merge_interactions(interaction, interaction_neg)
        elif ns_type == 'time':
            # # (u, i, ia) -> 1 as pos, then (u, i-, ia) & (u, i, ia+) as neg
            # interaction_neg1 = self._neg_sampling_random(train_set, full_negs=True)
            # interaction_neg2 = self._neg_sampling_time(interaction, full_negs=True)
            # interaction_ = self._merge_interactions(interaction, interaction_neg1)
            # interaction_ = self._merge_interactions(interaction_, interaction)
            # interaction_ = self._merge_interactions(interaction_, interaction_neg2)
            # # (u, i, T-) and (u, i-, all T) as negatives
            interaction_neg = self._neg_sampling_time_based()
            interaction_ = self._merge_interactions(interaction, interaction_neg)
        elif ns_type == 'time-only':
            # # (u,i,t) -> 1, (u,i,t+) -> 0
            # interaction_['user'] = torch.cat((users, users), 0)
            # interaction_['item'] = torch.cat((items, items), 0)
            # num = interaction['num'] 
            # itemage_neg = (itemage + torch.randint(1, self.data.n_periods, size=(num, )).to(self.device)).clip(0, self.data.n_periods-1)
            # target_neg = (itemage == itemage_neg).int()
            # # target_neg = torch.zeros_like(targets).to(self.device)
            # interaction_['target'] = torch.cat((targets, target_neg), 0)
            # interaction_['itemage'] = torch.cat((itemage, itemage_neg), 0)
            # interaction_['num'] = num * 2
            interaction_neg = self._neg_sampling_time(interaction, full_negs=True)
            interaction_ = self._merge_interactions(interaction, interaction_neg)
        else:
            raise NotImplementedError('[ns_type] should be implemented.')
        print("#pos: %d, #neg: %d" % (interaction['num'], interaction_['num'] - interaction['num']))
        # exit(0)
        return self._shuffle_date(interaction_)


    def _train_epoch(self, interaction, shuffle_data=True):
        if shuffle_data:
            interaction = self._shuffle_date(interaction)
        user = interaction['user']
        item = interaction['item']
        target = interaction['target']
        itemage = interaction['itemage']
        
        # batch split
        num_batch = int(np.ceil(interaction['num'] * 1.0 / self.batch_size))
        start_idxs, end_idxs = [], []
        for i_batch in range(num_batch - 1):
            start_idxs.append(i_batch * self.batch_size)
            end_idxs.append((i_batch+1) * self.batch_size)
        start_idxs.append((num_batch - 1) * self.batch_size)
        end_idxs.append(interaction['num'])
        start_idxs = np.array(start_idxs, dtype=int)
        end_idxs = np.array(end_idxs, dtype=int)

        # train on batch
        total_loss = 0
        for i_batch in range(len(start_idxs)):
            start_idx, end_idx = start_idxs[i_batch], end_idxs[i_batch]
            self.optimizer.zero_grad()
            losses = self.model.calculate_loss({'user': user[start_idx:end_idx], \
                'item': item[start_idx:end_idx], 'target': target[start_idx:end_idx], \
                'itemage': itemage[start_idx:end_idx]})
            total_loss += losses.item()
            losses.backward()
            self.optimizer.step()
        return total_loss

    def _shuffle_date(self, interaction):
        # shuffle the data
        order = torch.randperm(interaction['num']).to(self.device)
        for key in interaction.keys():
            if key != 'num':
                value = interaction[key]
                interaction[key] = value[order]
        return interaction
    def _numpy2tensor(self, interaction):
        for k in interaction.keys():
            if k != 'num':
                interaction[k] = torch.from_numpy(interaction[k]).to(self.device)
        return interaction
    
    def fit(self, valid_data=None, verbose=True, saved=True, resampling=False):
        # if self.splitting == 'random':
        #     interaction = self._numpy2tensor(self.data.train_interactions)
        #     interaction_valid = self._numpy2tensor(self.data.valid_interactions)
        # else: # else, time-based splitting, as old-version: generate negs during training
        #     interaction_pos = self._data_pre(self.train)
        interaction = self._numpy2tensor(self.data.train_interactions)
        interaction_valid = self._numpy2tensor(self.data.valid_interactions)
        
        start = time()
        for epoch_idx in range(self.epochs):
            # if (self.splitting != 'random') and ((epoch_idx == 0) or (resampling)):
            #     interaction = self._neg_sampling(interaction_pos, self.train)
            train_loss = self._train_epoch(interaction)
            if (epoch_idx + 1) % 1 == 0: # evaluate on valid set
                # self.load_model()
                if self.splitting == 'random':
                    valid_results, valid_loss = self.evaluate(interaction_valid, samplings=1.0)
                else:
                    valid_results, valid_loss = self.evaluate(interaction)
                print("epoch %d, time-consumin: %f s, train-loss: %f, valid-loss: %f, \nresults on validset: %s" % (epoch_idx+1, time()-start, train_loss, valid_loss, str(valid_results)))
                self.best_valid_score, _, stop_flag, _ = self._early_stopping(valid_loss, self.best_valid_score, epoch_idx, 10, bigger=False)
                # print(self.best_valid_score, stop_flag, valid_loss)
                if stop_flag:
                    print("Finished training, best eval result in epoch %d" % epoch_idx)
                    break
                self.save_model(epoch_idx)
                start = time()
        if self.splitting != 'random':
            valid_results, valid_loss = self.evaluate(interaction_valid, samplings=1.0)
            print("results on fixed valid set: %s" % (str(valid_results)))
        return self.model

    @torch.no_grad()      
    def _eval_epoch(self, interaction):
        user = interaction['user']
        item = interaction['item']
        target = interaction['target']
        itemage = interaction['itemage']
        
        # batch split
        num_batch = math.ceil(interaction['num'] * 1.0 / self.batch_size)
        start_idxs, end_idxs = [], []
        for i_batch in range(num_batch - 1):
            start_idxs.append(i_batch * self.batch_size)
            end_idxs.append((i_batch+1) * self.batch_size)
        start_idxs.append((num_batch - 1) * self.batch_size)
        end_idxs.append(interaction['num'])
        start_idxs = np.array(start_idxs, dtype=int)
        end_idxs = np.array(end_idxs, dtype=int)

        # eval on batch
        total_loss = 0
        scores = []
        for i_batch in range(len(start_idxs)):
            start_idx, end_idx = start_idxs[i_batch], end_idxs[i_batch]
            interaction_batch = {'user': user[start_idx:end_idx], 'item': item[start_idx:end_idx], \
                'itemage':itemage[start_idx:end_idx], 'target':target[start_idx:end_idx]}
            losses = self.model.calculate_loss(interaction_batch)
            scores.append(self.model.predict(interaction_batch))
            total_loss += losses.item()
        scores = torch.cat(scores, 0)
        # scores.clip(min(target), max(target)) # clip the output scores
        return total_loss, scores
    
    def _interaction_split(self, interaction, ratio=0.1):
        num2 = int(interaction['num'] * ratio) # valid
        reorder = torch.randperm(interaction['num']).to(self.device)
        interaction1, interaction2 = {}, {}
        interaction1['num'], interaction2['num'] = interaction['num'] - num2, num2
        for k in interaction.keys():
            if k != 'num':
                v = interaction[k][reorder]
                interaction1[k] = v[:-num2]
                interaction2[k] = v[-num2:]
        return interaction1, interaction2

    @torch.no_grad()
    def evaluate(self, interaction=None, samplings=0.1):
        # # use training set (pos+negs), sample a small
        if interaction is not None:
            if samplings < 1.0:
                num = int(interaction['num'] * samplings)
                idx = np.random.choice(interaction['num'], num)
                interaction_ = {}
                interaction_['num'] = num
                interaction_['user'] = interaction['user'][idx]
                interaction_['item'] = interaction['item'][idx]
                interaction_['itemage'] = interaction['itemage'][idx]
                interaction_['target'] = interaction['target'][idx]
                interaction = interaction_
        else:
            # # generate negatives based on validset
            interaction_pos = self._data_pre(self.data.valid)
            interaction = self._neg_sampling(interaction_pos, self.data.valid)

        # # losses and results
        # losses = self.model.calculate_loss(interaction).item()
        # scores = self.model.predict(interaction)
        losses, scores = self._eval_epoch(interaction)

        # find the position of the target item
        targets = interaction['target']
        results = cal_op_metrics(scores.cpu().numpy(), targets.cpu().numpy(), w_sigmoid=False)
        return results, losses


class OPPT_Trainer(AbstractTrainer):
    '''
    Observed user preference prediction task (OPPT) is to predict the observed user preference, w.r.t $P(y_{u,i,t}, o_{u,i,t})$.
    the time information could be a lot of things: user age, season, workday/weekend, even itemage.
    Here we first use itemage, since the OPPT also uses itemage so it is easy to apply.
    '''
    def __init__(self, config, model, data):
        super(OPPT_Trainer, self).__init__(config, model, data)
        if config['debiasing']:
            self.debiasing = "_ips"
        else:
            self.debiasing = '_naive'
        self.saved_model_file = "./checkpoint_dir/OPPT_" + config['dataset'] + '_' + config['mode'] + self.debiasing
        print("model will be saved into:", self.saved_model_file)
        
        self.optimizer = config['optimizer']
        self.epochs = config['epochs']
        self.batch_size = int(config['batch_size'])
        self.n_users = self.data.n_users
        self.n_items = self.data.n_items
        self.n_periods = self.data.n_periods

        self.optimizer = self._build_optimizer()
        self.item_birthdate = torch.from_numpy(self.data._get_item_birthdate()).to(self.device)
        # negative sampling strategy, random or time (with temporal order)
        if 'ns_type' not in config:
            self.ns_type = 'random'
        else:
            self.ns_type = config['ns_type'].lower()
        print("Training with Negative-Sampling: %s based" % self.ns_type)
        # # np.random.seed only control the train-test-splitting
        # torch.manual_seed(517)
        
    def _data_pre(self, train):
        '''
        Different from OIPT, we only need the observed data rather than with negatives (unobservation indicators).
        Return: (u, i, temporal_content, p(o_{u,i,t}), y), y is one-hot vector
        '''
        uid_list = torch.from_numpy(np.array(train['UserId'].values, dtype=int)).to(self.device)
        iid_list = torch.from_numpy(np.array(train['ItemId'].values, dtype=int)).to(self.device)
        if 'ItemAge' not in train.columns:
            itemage = self.data.get_itemage(train['ItemId'], train['timestamp'])
            itemage = torch.from_numpy(np.array(itemage, dtype=int)).to(self.device)
        else:
            itemage = torch.from_numpy(np.array(train['ItemAge'].values, dtype=int)).to(self.device)
        # target = torch.from_numpy(np.where(train['rating'].values > 3, 1, 0).astype(int)).to(self.device) # rating>3 like=1, <=3 dislike=0
        target = torch.from_numpy(train['rating'].values).to(self.device)
        if 'ips' in self.debiasing:
            predOP = torch.from_numpy(np.array(train['predOP'].values, dtype=float)).to(self.device)
        # target = torch.from_numpy(np.array(train['rating'].values, dtype=int)-1).to(self.device)

        # train_itemage = self.get_itemage(self.train['ItemId'], self.train['timestamp'])
        # valid_itemage = self.get_itemage(self.valid['ItemId'], self.valid['timestamp'])
        # print("The distributions of p_T in train set:", '\n', '[', ', '.join([str(self.train[train_itemage == t]['rating'].mean()) for t in range(self.n_periods)]), ']')
        # print("The distributions of p_T in valid set:", '\n', '[', ', '.join([str(self.valid[valid_itemage == t]['rating'].mean()) for t in range(self.n_periods)]), ']')
        # print("The distributions of p_T in subset:", '\n', '[', ', '.join([str(train[test_itemage == t]['rating'].mean()) for t in range(self.n_periods)]), ']')
        # exit(0)

        interaction = {}
        interaction['user'] = uid_list
        interaction['item'] = iid_list
        interaction['target'] = target
        interaction['itemage'] = itemage
        if 'ips' in self.debiasing:
            interaction['predOP'] = predOP
        interaction['num'] = uid_list.size()[0]
        return interaction

    def _shuffle_date(self, interaction):
        # shuffle the data
        order = torch.randperm(interaction['num'])
        for key in interaction.keys():
            if key != 'num':
                value = interaction[key]
                interaction[key] = value[order]
        return interaction

    def _train_epoch(self, interaction):
        user = interaction['user']
        item = interaction['item']
        target = interaction['target']
        itemage = interaction['itemage']
        num = interaction['num']
        if 'ips' in self.debiasing:
            inv_predOP = torch.reciprocal(interaction['predOP'])
        # train on batch
        total_loss = 0
        start_idx = 0
        end_idx = start_idx + self.batch_size
        while start_idx < num:
            self.optimizer.zero_grad()
            losses = self.model.calculate_loss({'user': user[start_idx:end_idx], \
                'item': item[start_idx:end_idx], 'target': target[start_idx:end_idx], \
                'itemage': itemage[start_idx:end_idx]})
            # In task OPPT, the reduction of calculate_loss is none, then ...
            if 'ips' in self.debiasing:
                losses = torch.mul(losses, inv_predOP[start_idx:end_idx]).mean() # w/ P(O)
            else:
                losses = losses.mean() # /o P(O)
            total_loss += losses.item()
            losses.backward()
            self.optimizer.step()
            start_idx = end_idx
            end_idx += self.batch_size
        return total_loss

    def fit(self, valid_data=None, verbose=True, saved=True, resampling=True):
        # interaction = self._data_pre(self.data.train_full)
        # interaction, interaction_valid = self.train_valid_split(interaction, sampling=0.1)
        interaction = self._data_pre(self.data.train)
        interaction_valid = self._data_pre(self.data.valid)

        start = time()
        for epoch_idx in range(self.epochs):
            if resampling:
                interaction = self._shuffle_date(interaction)
            train_loss = self._train_epoch(interaction)
            if (epoch_idx + 1) % 1 == 0: # evaluate on valid set
                # _, interaction_valid = self.train_valid_split(interaction, sampling=0.25)
                valid_results, valid_loss = self.evaluate(interaction_valid)
                print("epoch %d, time-consumin: %f s, train-loss: %f, valid-loss: %f, \nresults on validset: %s" % (epoch_idx+1, time()-start, train_loss, valid_loss, str(valid_results)))
                self.best_valid_score, _, stop_flag, _ = self._early_stopping(valid_loss, self.best_valid_score, epoch_idx, 10, bigger=False)
                # print(self.best_valid_score, stop_flag, valid_loss)
                if stop_flag:
                    print("Finished training, best eval result in epoch %d" % epoch_idx)
                    break
                start = time()
            self.save_model(epoch_idx)
        return self.model

    def train_valid_split(self, interaction, sampling=0.1):
        num = interaction['num']
        # np.random.seed(2021)
        indices = np.random.uniform(size=num)
        train_idx = indices >= sampling
        valid_idx = np.invert(train_idx)
        num_train, num_valid = (train_idx).sum(), (valid_idx).sum()
        assert (num_train + num_valid) == num

        interaction_train, interaction_valid = {}, {}
        interaction_train['num'] = num_train
        interaction_valid['num'] = num_valid
        for k in interaction.keys():
            if k != 'num':
                value = interaction[k]
                interaction_train[k] = value[train_idx]
                interaction_valid[k] = value[valid_idx]
        return interaction_train, interaction_valid

    @torch.no_grad()
    def evaluate(self, interaction):
        losses, scores = self._eval_epoch(interaction)
         # find the position of the target item
        targets = interaction['target']
        # print(targets.size(), scores.size())
        assert scores.size() == targets.size()
        # results = cal_op_metrics(scores.cpu().numpy(), targets.cpu().numpy(), w_sigmoid=False)
        predOP = None
        if 'ips' in self.debiasing:
            predOP = interaction['predOP'].cpu().numpy() # np.reciprocal(interaction['predOP'].cpu().numpy()) # IPS 
        results = cal_ratpred_metrics(scores.cpu().numpy(), targets.cpu().numpy(), predOP = predOP, users = interaction['user'].cpu().numpy())
        return results, losses

    @torch.no_grad()      
    def _eval_epoch(self, interaction):
        user = interaction['user']
        item = interaction['item']
        target = interaction['target']
        itemage = interaction['itemage']
        if 'ips' in self.debiasing:
            inv_predOP = torch.reciprocal(interaction['predOP'])
        num = interaction['num']

        start_idx = 0
        end_idx = start_idx + self.batch_size
        total_loss = 0
        scores = []
        while start_idx < num:
            interaction_batch = {'user': user[start_idx:end_idx], \
                'item': item[start_idx:end_idx], 'target': target[start_idx:end_idx], 'itemage': itemage[start_idx:end_idx]}
            losses = self.model.calculate_loss(interaction_batch)
            if 'ips' in self.debiasing:
                losses = torch.mul(losses, inv_predOP[start_idx: end_idx]).mean() # w/ P(O)
            else:
                losses = losses.mean() # /o P(O)
            total_loss += losses.item()
            scores.append(self.model.predict(interaction_batch))
            start_idx = end_idx
            end_idx += self.batch_size
            # print(scores[-1].size())
        scores = torch.cat(scores, 0)
        # scores.clip(min(target), max(target)) # clip the output scores
        # scores = scores * 0 + 3.514037

        # preds_ = np.stack((target.cpu().numpy(), scores.cpu().numpy(), np.zeros_like(scores.cpu().numpy())+3.514037),axis=1)
        # np.savetxt('preds_onValid.csv', preds_)
        return total_loss, scores
    

class TART_Trainer(AbstractTrainer):
    '''
    Time-Aware Recommendation Task (TART) is to predict the ratings of users and recommend items to users.
    The model w/o debiasing only learns and evaluates on the simuated dataset, where we generate:
    data.train/valid: [UserId,ItemId,itemage,rating,predOP], and data.test: [UserId,ItemId,itemage,rating]
    '''
    def __init__(self, config, model, data):
        super(TART_Trainer, self).__init__(config, model, data)
        if config['debiasing']:
            self.debiasing = "_ips"
        else:
            self.debiasing = '_naive'
        self.saved_model_file = "./checkpoint_dir/TART_" + config['dataset'] + '_' + config['mode'] + self.debiasing
        print("model will be saved into:", self.saved_model_file)
        
        self.optimizer = config['optimizer']
        self.epochs = config['epochs']
        self.batch_size = int(config['batch_size'])
        self.n_users = self.data.n_users
        self.n_items = self.data.n_items
        self.n_periods = self.data.n_periods

        self.optimizer = self._build_optimizer()
        
    def _data_pre(self, train):
        '''
        Different from OIPT, we only need the observed data rather than with negatives (unobservation indicators).
        Return: (u, i, temporal_content, p(o_{u,i,t}), y), y is one-hot vector
        '''
        interaction = {}
        interaction['user'] = torch.from_numpy(np.array(train['UserId'].values, dtype=int)).to(self.device)
        interaction['item'] = torch.from_numpy(np.array(train['ItemId'].values, dtype=int)).to(self.device)
        interaction['itemage'] = torch.from_numpy(np.array(train['itemage'].values, dtype=int)).to(self.device)
        interaction['target'] = torch.from_numpy(train['rating'].values).to(self.device)
        interaction['predOP'] = torch.from_numpy(np.array(train['predOP'].values, dtype=float)).to(self.device)
        interaction['num'] = len(train)
        return interaction

    def _shuffle_date(self, interaction):
        # shuffle the data
        order = torch.randperm(interaction['num'])
        for key in interaction.keys():
            if key != 'num':
                value = interaction[key]
                interaction[key] = value[order]
        return interaction

    def _train_epoch(self, interaction):
        user = interaction['user']
        item = interaction['item']
        target = interaction['target']
        itemage = interaction['itemage']
        num = interaction['num']
        if 'ips' in self.debiasing:
            inv_predOP = torch.reciprocal(interaction['predOP'])
            # print("!!!! Using SNIPS !!!!")
            # inv_predOP = inv_predOP / inv_predOP.mean()
        # train on batch
        total_loss = 0
        start_idx = 0
        end_idx = start_idx + self.batch_size
        while start_idx < num:
            self.optimizer.zero_grad()
            losses = self.model.calculate_loss({'user': user[start_idx:end_idx], \
                'item': item[start_idx:end_idx], 'target': target[start_idx:end_idx], \
                'itemage': itemage[start_idx:end_idx]})
            # In task TART, the reduction of calculate_loss is none, then ...
            if 'ips' in self.debiasing:
                losses = torch.mul(losses, inv_predOP[start_idx:end_idx]).mean() #.sum() / (self.n_users * self.n_items) # w/ P(O)
            else:
                losses = losses.mean() # /o P(O)
            total_loss += losses.item()
            losses.backward()
            self.optimizer.step()
            start_idx = end_idx
            end_idx += self.batch_size
        return total_loss

    def fit(self, valid_data=None, verbose=True, saved=True, resampling=True):
        interaction = self._data_pre(self.data.train)
        interaction_valid = self._data_pre(self.data.valid)

        start = time()
        for epoch_idx in range(self.epochs):
            if resampling:
                interaction = self._shuffle_date(interaction)
            train_loss = self._train_epoch(interaction)
            if (epoch_idx + 1) % 1 == 0: # evaluate on valid set
                valid_results, valid_loss = self.evaluate(interaction_valid)
                print("epoch %d, time-consumin: %f s, train-loss: %f, valid-loss: %f, \nresults on validset: %s" % (epoch_idx+1, time()-start, train_loss, valid_loss, str(valid_results)))
                self.best_valid_score, _, stop_flag, _ = self._early_stopping(valid_loss, self.best_valid_score, epoch_idx, 10, bigger=False)
                if stop_flag:
                    print("Finished training, best eval result in epoch %d" % epoch_idx)
                    break
                start = time()
            self.save_model(epoch_idx)
        return self.model

    @torch.no_grad()
    def evaluate(self, interaction):
        losses, scores = self._eval_epoch(interaction)
        targets = interaction['target']
        assert scores.size() == targets.size()
        # results = cal_op_metrics(scores.cpu().numpy(), targets.cpu().numpy(), w_sigmoid=False)
        predOP = None
        if 'ips' in self.debiasing:
            predOP = interaction['predOP'].cpu().numpy() # np.reciprocal(interaction['predOP'].cpu().numpy()) # IPS 
        results = cal_ratpred_metrics(scores.cpu().numpy(), targets.cpu().numpy(), predOP = predOP, users = interaction['user'].cpu().numpy())
        return results, losses

    @torch.no_grad()      
    def _eval_epoch(self, interaction):
        user = interaction['user']
        item = interaction['item']
        target = interaction['target']
        itemage = interaction['itemage']
        if 'ips' in self.debiasing:
            inv_predOP = torch.reciprocal(interaction['predOP'])
            # # print("!!!! Using SNIPS !!!!")
            # inv_predOP = inv_predOP / inv_predOP.mean()
        num = interaction['num']

        start_idx = 0
        end_idx = start_idx + self.batch_size
        total_loss = 0
        scores = []
        while start_idx < num:
            interaction_batch = {'user': user[start_idx:end_idx], \
                'item': item[start_idx:end_idx], 'target': target[start_idx:end_idx], 'itemage': itemage[start_idx:end_idx]}
            losses = self.model.calculate_loss(interaction_batch)
            if 'ips' in self.debiasing:
                losses = torch.mul(losses, inv_predOP[start_idx: end_idx]).mean() # w/ P(O)
            else:
                losses = losses.mean() # /o P(O)
            total_loss += losses.item()
            scores.append(self.model.predict(interaction_batch))
            start_idx = end_idx
            end_idx += self.batch_size
            # print(scores[-1].size())
        scores = torch.cat(scores, 0)
        return total_loss, scores