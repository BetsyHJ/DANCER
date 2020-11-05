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
from evaluator.evaluator import calculate_metrics

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
        self.learner = 'Adam'
        self.learning_rate = 1e-3

    def fit(self):
        r"""Train the model based on the train data.
        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.
        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    def __init__(self, config, model, data):
        super(Trainer, self).__init__(config, model, data)

        self.optimizer = config['optimizer']
        self.saved_model_file = "./checkpoint_dir/ml-100k" # config['checkpoint_dir']
        self.epochs = 100 # config['epochs']
        self.batch_size = 512
        self.max_item_list_len = 20 # config['max_item_list_len']

        self.optimizer = self._build_optimizer()

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
        ctr_target = torch.from_numpy(np.array(ctr_target, dtype=int)).to(self.device)
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
            u_n_items = len(u_items) - 1
            target.append(u_items[-1])
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
        # ctr_target = torch.from_numpy(np.array(ctr_target, dtype=int)).to(self.device)
        interaction = {}
        interaction['seq'] = seq_list
        interaction['seq_len'] = seq_len
        interaction['target'] = target
        # interaction['ctr'] = ctr_target
        assert seq_list.size()[0] == seq_len.size()[0]
        interaction['num'] = seq_list.size()[0]
        return interaction

    def _train_epoch(self, interaction, shuffle_data=True):
        if shuffle_data:
            # shuffle the data
            order = np.arange(interaction['num'], dtype=int)
            np.random.shuffle(order)
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
            if (epoch_idx + 1) % 10 == 0: # evaluate on valid set
                self.save_model(epoch_idx)
                # self.load_model()
                results = self.evaluate()
                print("epoch %d, time-consumin: %f s, train-loss: %f, \nresults on validset: %s" % (epoch_idx+1, time()-start, train_loss, str(results)))
                start = time()
                
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
        scores = self.model.full_sort_predict(interaction)
        # find the position of the target item
        targets = interaction['target']
        target_scores = torch.gather(scores, 1, targets.view(-1, 1)) # [B 1]
        target_pos = (scores >= target_scores).sum(-1) # [B]
        results = calculate_metrics(target_pos.cpu().numpy())
        return results
        


