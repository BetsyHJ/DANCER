import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_
from torch.nn.parameter import Parameter

class TMF(nn.Module):
    '''
    Time-aware matrix factorization
    Cite: Collaborative filtering with temporal dynamics
    We only consider q_i(t) here when modeling r_{u,i,t}
    '''
    def __init__(self, config, data, debiasing=False):
        super(TMF, self).__init__()

        # load parameter info
        self.debiasing = debiasing
        self.embedding_size = int(config['embedding_size'])
        self.loss_type = config['loss_type']
        # self.lr_decay_step = int(config['lr_decay_step'])
        self.batch_size = int(config['batch_size'])
        self.n_items = data.n_items 
        self.n_periods = data.n_periods 
        self.n_users = data.n_users

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.item_Dyn_embedding = nn.Embedding(self.n_items * self.n_periods, self.embedding_size)
        # self.item_Dyn_embedding = [] # [T N D]
        # for t in range(self.n_periods):
        #     self.item_Dyn_embedding.append(nn.Embedding(self.n_items, self.embedding_size))
        
        if self.loss_type.upper() == 'CE':
            if self.debiasing:
                self.loss_fct = nn.CrossEntropyLoss(reduction='none')
            else:
                self.loss_fct = nn.CrossEntropyLoss()
        elif self.loss_type.upper() == 'MSE':
            self.loss_fct = nn.MSELoss()
        elif self.loss_type.upper() == 'NLL':
            # self.loss_fct = nn.NLLLoss(reduction='none')
            self.loss_fct = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['CE', 'MSE', 'NLL']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(self.gru_layers.weight_hh_l0)
            xavier_uniform_(self.gru_layers.weight_ih_l0)
        # elif isinstance(module, nn.Linear):
        #     xavier_uniform_(module.weight)
    
    def _gather_indexes(self, output, gather_index):
        """Gathers the vectors at the spexific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def get_item_embedding(self, item, itemage): # [B], [B]
        idx = item * self.n_periods + itemage
        return self.item_embedding(item) + self.item_Dyn_embedding(idx)

    def forward(self, user, item, itemage):
        user_e = self.user_embedding(user)
        item_e = self.get_item_embedding(item, itemage)
        return torch.mul(user_e, item_e).sum(-1).float() # [B, D] -> [B]
        
    def calculate_loss(self, interaction):
        user = interaction['user']
        item = interaction['item']
        itemage = interaction['itemage']
        pred = self.forward(user, item, itemage)
        target = interaction['target'].float()
        loss = self.loss_fct(pred, target)
        # if self.debiasing:
        #     ctr = torch.reciprocal(interaction['ctr']) # [B]
        #     loss = torch.mul(loss, ctr).sum() # [B] -> [1]
        return loss

    def predict(self, interaction):
        user = interaction['user']
        item = interaction['item']
        itemage = interaction['itemage']
        pred = self.forward(user, item, itemage)
        return pred

    def full_sort_predict(self, interaction):
        user = interaction['user']
        test_items_emb = self.item_embedding.weight.view(self.n_items, 1, self.embedding_size) + \
                self.item_Dyn_embedding.weight.view(self.n_items, self.n_periods, self.embedding_size) # [N D] + [N T D] -> [N T D]
        test_items_emb = test_items_emb.view(self.n_items * self.n_periods, self.embedding_size) #[N*T D]
        scores = torch.matmul(self.user_embedding(user), test_items_emb.transpose(0, 1))  # [B D], [D N*T] -> [B N*T]
        return scores.view(-1, self.n_items, self.n_periods) # [B N T]


class TMF_fast(nn.Module):
    '''
    Modified time-aware matrix factorization
    r_{u,i} = p_u * q_i
    f_{u,i,t} = sigmoid(w * r_{u,i}*t + b)
    '''
    def __init__(self, config, data, debiasing=False):
        super(TMF_fast, self).__init__()
        
        print("********* Using TMF-fast **********")
        # load parameter info
        self.debiasing = debiasing
        self.embedding_size = int(config['embedding_size'])
        self.loss_type = config['loss_type']
        # self.lr_decay_step = int(config['lr_decay_step'])
        self.batch_size = int(config['batch_size'])
        self.n_items = data.n_items 
        self.n_periods = data.n_periods 
        self.n_users = data.n_users

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.dense = nn.Linear(1, 1)
        # self.b = Parameter(torch.Tensor(1))
        self.w = Parameter(torch.Tensor(1))
        
        if self.loss_type.upper() == 'CE':
            if self.debiasing:
                self.loss_fct = nn.CrossEntropyLoss(reduction='none')
            else:
                self.loss_fct = nn.CrossEntropyLoss()
        elif self.loss_type.upper() == 'MSE':
            self.loss_fct = nn.MSELoss()
        elif self.loss_type.upper() == 'NLL':
            # self.loss_fct = nn.NLLLoss(reduction='none')
            self.loss_fct = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['CE', 'MSE', 'NLL']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(self.gru_layers.weight_hh_l0)
            xavier_uniform_(self.gru_layers.weight_ih_l0)
        # elif isinstance(module, nn.Linear):
        #     xavier_uniform_(module.weight)
    
    def _gather_indexes(self, output, gather_index):
        """Gathers the vectors at the spexific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def forward(self, user, item, itemage):
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        r_ui = torch.mul(user_e, item_e).sum(-1).float() # [B D] -> [B]
        # # W * p1 + b
        # f_uit = self.dense(torch.mul(r_ui, itemage).unsqueeze(1)).squeeze().float() # [B 1]
        # # p1 + b
        f_uit = torch.mul(torch.mul(r_ui, itemage), self.w) # + self.b
        return f_uit # [B, D] -> [B]
        
    def calculate_loss(self, interaction):
        user = interaction['user']
        item = interaction['item']
        itemage = interaction['itemage']
        pred = self.forward(user, item, itemage)
        target = interaction['target'].float()
        loss = self.loss_fct(pred, target)
        # if self.debiasing:
        #     ctr = torch.reciprocal(interaction['ctr']) # [B]
        #     loss = torch.mul(loss, ctr).sum() # [B] -> [1]
        return loss

    def predict(self, interaction):
        user = interaction['user']
        item = interaction['item']
        itemage = interaction['itemage']
        pred = self.forward(user, item, itemage)
        return pred

    def full_sort_predict(self, interaction):
        user = interaction['user']
        test_items_emb = self.item_embedding.weight.view(self.n_items, 1, self.embedding_size) + \
                self.item_Dyn_embedding.weight.view(self.n_items, self.n_periods, self.embedding_size) # [N D] + [N T D] -> [N T D]
        test_items_emb = test_items_emb.view(self.n_items * self.n_periods, self.embedding_size) #[N*T D]
        scores = torch.matmul(self.user_embedding(user), test_items_emb.transpose(0, 1))  # [B D], [D N*T] -> [B N*T]
        return scores.view(-1, self.n_items, self.n_periods) # [B N T]

class TMF_fast_variety(TMF_fast):
    '''
    Modified time-aware matrix factorization
    s_{u,i} = p_u * q_i
    previous: f_{u,i,t} = sigmoid(w * r_{u,i}*T + b), w & b same for all (u,i) pair, T is itemage.
    current:  f_{u,i,t} = sigmoid(s1 * T + s2)
    '''
    def __init__(self, config, data, debiasing=False):
        super(TMF_fast_variety, self).__init__(config, data, debiasing)
        # del useless params
        self.user_embedding, self.item_embedding = None, None
        self.dense = None
        self.W1 = Parameter(torch.Tensor(1))
        self.W2 = Parameter(torch.Tensor(1))
        self.b = Parameter(torch.Tensor(1))

        # define layers and loss
        self.n_factors = 2
        if self.n_factors == 2:
            # print("TMF_fast_variety2: sigmoid(p1 * T1 + p2)")
            print("TMF_fast_variety2': sigmoid(W1 * p1 * T1 + W2 * p2 + b)")
        else:
            raise NotImplementedError("Make sure 'n_factors' in [2]!")
        self.user_embedding = nn.Embedding(self.n_users * self.n_factors, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items * self.n_factors, self.embedding_size)
        # self.apply(self._init_weights)

    def forward(self, user, item, itemage):
        prob_scores = [] # s1, s2, ...
        for i in range(self.n_factors):
            prob_scores.append(torch.mul(self.user_embedding(self.n_users * i + user), \
                    self.item_embedding(self.n_items * i + item)).sum(-1).float()) # [B]
        # # p1 * T1 + p2
        # output = torch.mul(prob_scores[0], itemage) + prob_scores[1]
        # # W1 * p1 * T1 + W2 * p2 + b
        output = self.W1 * torch.mul(prob_scores[0], itemage) + self.W2 * prob_scores[1] + self.b
        return output.squeeze().float()

