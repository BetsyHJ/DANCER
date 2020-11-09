import numpy as np
import torch

# metric definition
# target position starts from 1 to m.
def hit_(target_pos, K):
    return (target_pos <= K).mean()
def mrr_(target_pos, K):
    mrr = 1.0 / target_pos
    mrr = np.where(target_pos <= K, mrr, 0.0)
    return mrr.mean()
def ndcg_(target_pos, K):
    result = np.log(2.) / np.log(1.0 + target_pos)
    result = np.where(target_pos <= K, result, 0.0)
    return result.mean()


def calculate_metrics(target_pos):
    results = {}
    K = 10
    results['mrr'] = round(mrr_(target_pos, K), 3)
    results['hit@'+str(K)] = round(hit_(target_pos, K), 3)
    results['ndcg@'+str(K)] = round(ndcg_(target_pos, K), 3)
    return results

# def out_results(results):
#     output = ''
#     for key in results.key():

class AbstractEvaluator(object):
    """:class:`AbstractEvaluator` is an abstract object which supports
    the evaluation of the model. It is called by :class:`Trainer`.
    Note:       
        If you want to inherit this class and implement your own evalautor class, 
        you must implement the following functions.
    Args:
        config (Config): The config of evaluator.
    """

    def __init__(self, config, model, data):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.config = config
        self.data = data

    def _check_args(self):
        """check the correct of the setting"""
        raise NotImplementedError

    def collect(self):
        """get the intermediate results for each batch, it is called at the end of each batch"""
        raise NotImplementedError

    def evaluate(self):
        """calculate the metrics of all batches, it is called at the end of each epoch"""
        raise NotImplementedError

    def metrics_info(self):
        """get metrics result"""
        raise NotImplementedError

    def _calculate_metrics(self):
        """ to calculate the metrics"""
        raise NotImplementedError

class Evaluator(AbstractEvaluator):
    def __init__(self, config, model, data):
        super(Evaluator, self).__init__(config, model, data)
        self.max_item_list_len = 20 # config['max_item_list_len']

    def _data_pre_fullseq(self, train_full, test):
        train = train_full
        n_items = self.data.n_items
        uid_list, seq_list, seq_len, ctr_target, target = [], [], [], [], []
        u_target = dict(zip(test['UserId'], test['ItemId']))
        for u, group in train.groupby(['UserId']):
            u_ratings = group.sort_values(['timestamp'], ascending=True)
            u_items = u_ratings['ItemId'].values
            u_n_items = len(u_items)
            target.append(u_target[u])
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
    
    def evaluate(self):
        interaction = self._data_pre_fullseq(self.data.train_full, self.data.test)
        # results
        scores = self.model.full_sort_predict(interaction)
        targets = interaction['target']
        target_scores = torch.gather(scores, 1, targets.view(-1, 1)) # [B 1]
        target_pos = (scores >= target_scores).sum(-1) # [B]
        results = calculate_metrics(target_pos.cpu().numpy())
        print("results on testset: %s" % (str(results)))
        return results
    
    # def metrics_info(self, results):
        