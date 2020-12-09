import numpy as np
import torch

# metric definition
# target position starts from 1 to m.
def hit_(target_pos, K):
    return (target_pos <= K)
def mrr_(target_pos, K):
    mrr = 1.0 / target_pos
    mrr = np.where(target_pos <= K, mrr, 0.0)
    return mrr
def ndcg_(target_pos, K):
    result = np.log(2.) / np.log(1.0 + target_pos)
    result = np.where(target_pos <= K, result, 0.0)
    return result


def calculate_metrics(target_pos, weight=None, normalization='snips', threshold=1e-3):
    results = {}
    K = 10
    if weight is None:
        print("evaluation on standard metrics")
        results['mrr@'+str(K)] = round(mrr_(target_pos, K).mean(), 8)
        results['hit@'+str(K)] = round(hit_(target_pos, K).mean(), 8)
        results['ndcg@'+str(K)] = round(ndcg_(target_pos, K).mean(), 8)
    elif normalization.lower() == "snips":
        print("evaluation on snips metrics")
        weight_inverse = np.reciprocal(weight)
        weight_ = weight_inverse / weight_inverse.sum() # normalization based on users, since one target per user
        results['mrr@'+str(K)] = round((mrr_(target_pos, K) * weight_).sum(), 8)
        results['hit@'+str(K)] = round((hit_(target_pos, K) * weight_).sum(), 8)
        results['ndcg@'+str(K)] = round((ndcg_(target_pos, K) * weight_).sum(), 8)
    elif normalization == 'pop':
        # threshold = 1.0e-3
        condition = (weight > threshold)
        print("evaluation on pop items with ctr bigger than", threshold, ", #ui_pairs", condition.sum())
        results['mrr@'+str(K)] = round((mrr_(target_pos, K)[condition]).mean(), 3)
        results['hit@'+str(K)] = round((hit_(target_pos, K)[condition]).mean(), 3)
        results['ndcg@'+str(K)] = round((ndcg_(target_pos, K)[condition]).mean(), 3)
    elif normalization == 'unpop':
        # threshold = 1.0e-3
        condition = (weight < threshold)
        print("evaluation on un-pop items with ctr smaller than", threshold, ", #ui_pairs", condition.sum())
        results['mrr@'+str(K)] = round((mrr_(target_pos, K)[condition]).mean(), 3)
        results['hit@'+str(K)] = round((hit_(target_pos, K)[condition]).mean(), 3)
        results['ndcg@'+str(K)] = round((ndcg_(target_pos, K)[condition]).mean(), 3)

    else:
        raise NotImplementedError
    return results

def _mse(pred, true):
    return np.mean((true - pred)**2)
def _mae(pred, true):
    return np.absolute(true - pred).mean()
def _acc(pred, true):
    pred = np.round(pred).astype(int)
    return (pred == true).mean()
# def _perplexity()

def cal_ratpred_metrics(score, target):
    results = {}
    results['mse'] = _mse(score, target)
    results['mae'] = _mae(score, target)
    results['acc'] = _acc(score, target)
    return results

def _nll(score, true):
    nll_score = true * np.log(score) + (1.0-true) * np.log(1-score) 
    return - nll_score.mean()

def cal_op_metrics(score, target):
    score = 1.0 / (1.0 + np.exp(-score)) # sigmoid(score)
    results = {}
    results['acc'] = _acc(score, target)
    results['nll'] = _nll(score, target)
    return results

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
        u_ctr = dict(zip(test['UserId'], test['ctr']))
        for u, group in train.groupby(['UserId']):
            u_ratings = group.sort_values(['timestamp'], ascending=True)
            u_items = u_ratings['ItemId'].values
            u_n_items = len(u_items)
            target.append(u_target[u])
            ctr_target.append(u_ctr[u])
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
        # ctr_target = torch.from_numpy(np.array(ctr_target, dtype=float)).to(self.device)
        ctr_target = np.array(ctr_target, dtype=float)
        interaction = {}
        interaction['seq'] = seq_list
        interaction['seq_len'] = seq_len
        interaction['target'] = target
        # interaction['ctr'] = ctr_target
        assert seq_list.size()[0] == seq_len.size()[0]
        interaction['num'] = seq_list.size()[0]
        return interaction, ctr_target
    
    def evaluate(self, ub='false', threshold=1e-3):
        ''' ub: unbiased evaluator'''
        interaction, ctr_target = self._data_pre_fullseq(self.data.train_full, self.data.test)
        # results
        scores = self.model.full_sort_predict(interaction)
        targets = interaction['target']
        target_scores = torch.gather(scores, 1, targets.view(-1, 1)) # [B 1]
        target_pos = (scores >= target_scores).sum(-1) # [B]
        if ub.lower() in ['snips', 'pop', 'unpop']:
            results = calculate_metrics(target_pos.cpu().numpy(), weight=ctr_target, normalization=ub, threshold=threshold)
        else:
            results = calculate_metrics(target_pos.cpu().numpy())
        print("results on testset: %s" % (str(results)))
        return results
    
    # def metrics_info(self, results):
        
class RatPred_Evaluator(AbstractEvaluator):
    def __init__(self, config, model, data):
        super(RatPred_Evaluator, self).__init__(config, model, data)

    def _data_pre(self, test):
        uid_list = torch.from_numpy(np.array(test['UserId'].values, dtype=int)).to(self.device)
        iid_list = torch.from_numpy(np.array(test['ItemId'].values, dtype=int)).to(self.device)
        target = torch.from_numpy(np.array(test['rating'].values, dtype=float)).to(self.device)
        itemage = torch.from_numpy(np.array(test['ItemAge_month'].values, dtype=int)).to(self.device)
        
        interaction = {}
        interaction['user'] = uid_list
        interaction['item'] = iid_list
        interaction['target'] = target
        interaction['itemage'] = itemage

        interaction['num'] = uid_list.size()[0]
        return interaction
    
    @torch.no_grad() 
    def evaluate(self, ub='false', threshold=1e-3):
        ''' ub: unbiased evaluator'''
class RatPred_Evaluator(AbstractEvaluator):
    def __init__(self, config, model, data):
        super(RatPred_Evaluator, self).__init__(config, model, data)

    def _data_pre(self, test):
        uid_list = torch.from_numpy(np.array(test['UserId'].values, dtype=int)).to(self.device)
        iid_list = torch.from_numpy(np.array(test['ItemId'].values, dtype=int)).to(self.device)
        target = torch.from_numpy(np.array(test['rating'].values, dtype=float)).to(self.device)
        itemage = torch.from_numpy(np.array(test['ItemAge_month'].values, dtype=int)).to(self.device)
        
        interaction = {}
        interaction['user'] = uid_list
        interaction['item'] = iid_list
        interaction['target'] = target
        interaction['itemage'] = itemage

        interaction['num'] = uid_list.size()[0]
        return interaction
    
    @torch.no_grad() 
    def evaluate(self, ub='false', threshold=1e-3):
        ''' ub: unbiased evaluator'''
        interaction = self._data_pre(self.data.test)
        # results
        scores = self.model.predict(interaction)
        targets = interaction['target']
        results = cal_ratpred_metrics(scores.cpu().numpy(), targets.cpu().numpy())
        print("results on testset: %s" % (str(results)))
        return results
    
    # def metrics_info(self, results):
        

class OP_Evaluator(AbstractEvaluator):
    def __init__(self, config, model, data):
        super(OP_Evaluator, self).__init__(config, model, data)
    
    def _data_pre(self, test):
        uid_list = torch.from_numpy(np.array(test['UserId'].values, dtype=int)).to(self.device)
        iid_list = torch.from_numpy(np.array(test['ItemId'].values, dtype=int)).to(self.device)
        target = torch.ones_like(iid_list).to(self.device)
        itemage = torch.from_numpy(np.array(test['ItemAge_month'].values, dtype=int)).to(self.device)
        timestamp = torch.from_numpy(np.array(test['timestamp'].values, dtype=int)).to(self.device)

        interaction = {}
        interaction['user'] = uid_list
        interaction['item'] = iid_list
        interaction['target'] = target
        interaction['itemage'] = itemage
        interaction['timestamp'] = timestamp
        interaction['num'] = uid_list.size()[0]
        return interaction

    @torch.no_grad() 
    def evaluate(self, ub='false', threshold=1e-3):
        ''' ub: unbiased evaluator'''
        interaction = self._data_pre(self.data.test)
        # results
        scores = self.model.predict(interaction)
        targets = interaction['target']
        results = cal_op_metrics(scores.cpu().numpy(), targets.cpu().numpy())
        print("results on testset: %s" % (str(results)))
        return results
    
    # def metrics_info(self, results):
        