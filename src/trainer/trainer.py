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
        self.learner = config['optimizer']
        self.learning_rate = float(config['learning_rate'])

        # 
        self.best_valid_score = None
    
    def _build_optimizer(self):
        r"""Init the Optimizer
        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
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
        interaction = self._data_pre(self.data.train)
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

    def save_model(self, epoch):
        state = {'net': self.model.state_dict(), 'optimizer':self.optimizer.state_dict(), 'epoch':epoch}
        torch.save(state, self.saved_model_file)
    
    def load_model(self):
        checkpoint = torch.load(self.saved_model_file)
        self.model.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
    
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
        interaction = self._data_pre(self.data.train)
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

    def save_model(self, epoch):
        state = {'net': self.model.state_dict(), 'optimizer':self.optimizer.state_dict(), 'epoch':epoch}
        torch.save(state, self.saved_model_file)
    
    def load_model(self):
        checkpoint = torch.load(self.saved_model_file)
        self.model.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

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

        self.optimizer = config['optimizer']
        self.saved_model_file = "./checkpoint_dir/" + config['dataset'] + '_' + config['mode'] + '_ObsPred'
        print("model will be saved into:", self.saved_model_file)
        self.epochs = config['epochs']
        self.batch_size = int(config['batch_size'])
        self.n_items = self.data.n_items

        self.optimizer = self._build_optimizer()
        self.item_birthdate = torch.from_numpy(self.data._get_item_birthdate()).to(self.device)
        

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

    def _neg_sampling(self, interaction):
        interaction_ = {}
        users = interaction['user']
        items = interaction['item']
        targets = interaction['target']
        timestamp = interaction['timestamp']
        itemage = interaction['itemage']

        # generate data for negs
        interaction_['user'] = torch.cat((users, users), 0)
        negs = torch.randint(self.n_items, size=(interaction['num'],)).to(self.device)
        interaction_['item'] = torch.cat((items, negs), 0)
        target_neg = (items == negs).int()
        interaction_['target'] = torch.cat((targets, target_neg), 0)
        itemage_neg = self.data.get_itemage(items, timestamp, self.item_birthdate)
        # itemage_neg = ((timestamp - self.item_birthdate[items]) * 1.0 / (30*24*60*60)).int().clip(0, self.data.n_months - 1) # unit: month
        interaction_['itemage'] = torch.cat((itemage, itemage_neg), 0)
        interaction_['num'] = interaction['num'] * 2
        return self._shuffle_date(interaction_)


    def _train_epoch(self, interaction, shuffle_data=True):
        if shuffle_data:
            interaction = self._shuffle_date(interaction)
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
        order = torch.randperm(interaction['num'])
        for key in ['user', 'item', 'itemage', 'target']:
            value = interaction[key]
            interaction[key] = value[order]
        return interaction
    
    def fit(self, valid_data=None, verbose=True, saved=True, resampling=True):
        interaction_pos = self._data_pre(self.data.train)
        start = time()
        for epoch_idx in range(self.epochs):
            if (epoch_idx == 0) or (resampling):
                interaction = self._neg_sampling(interaction_pos) 
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

        for i_batch in range(len(start_idxs)):
            start_idx, end_idx = start_idxs[i_batch], end_idxs[i_batch]
            self.optimizer.zero_grad()
            losses = self.model.calculate_loss({'user': user[start_idx:end_idx], \
                'item': item[start_idx:end_idx], 'itemage':itemage[start_idx:end_idx]})
            total_loss += losses.item()
            losses.backward()
            self.optimizer.step()
        return total_loss

    def save_model(self, epoch):
        state = {'net': self.model.state_dict(), 'optimizer':self.optimizer.state_dict(), 'epoch':epoch}
        torch.save(state, self.saved_model_file)
    
    def load_model(self):
        checkpoint = torch.load(self.saved_model_file)
        self.model.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
    
    @torch.no_grad()
    def evaluate(self):
        interaction_pos = self._data_pre(self.data.valid)
        interaction = self._neg_sampling(interaction_pos)
        # losses
        losses = self.model.calculate_loss(interaction).item()
        # results
        scores = self.model.predict(interaction)
        # find the position of the target item
        targets = interaction['target']
        results = cal_op_metrics(scores.cpu().numpy(), targets.cpu().numpy())
        return results, losses