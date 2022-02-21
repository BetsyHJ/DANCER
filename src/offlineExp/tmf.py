import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
from torch.nn.parameter import Parameter

class TMF(nn.Module):
    '''
    Time-aware matrix factorization
    Cite: Collaborative filtering with temporal dynamics
    We only consider q_i(t) here when modeling r_{u,i,t}
    '''
    def __init__(self, config, data, debiasing=False, output_dim=2):
        super(TMF, self).__init__()

        self.task = config['task']
        # load parameter info
        self.debiasing = debiasing
        self.embedding_size = int(config['embedding_size'])
        self.loss_type = config['loss_type']
        # self.lr_decay_step = int(config['lr_decay_step'])
        self.batch_size = int(config['batch_size'])
        self.n_items = data.n_items 
        self.n_periods = data.n_periods 
        self.n_users = data.n_users
        self.output_dim = output_dim

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        # self.item_Dyn_embedding = nn.Embedding(self.n_items * self.n_periods, self.embedding_size)
        self.user_Dyn_embedding = nn.Embedding(self.n_users * self.n_periods, self.embedding_size)

        self.b_u = nn.Embedding(self.n_users, 1)
        self.b_i = nn.Embedding(self.n_items, 1)
        self.b = Parameter(torch.Tensor(1))
        self.global_T = nn.Embedding(self.n_periods, 1)

        # self.item_Dyn_embedding = [] # [T N D]
        # for t in range(self.n_periods):
        #     self.item_Dyn_embedding.append(nn.Embedding(self.n_items, self.embedding_size))

        self.m = None
        reduction = 'mean'
        if self.task == 'OPPT':
            reduction = 'none'
        if self.loss_type.upper() == 'CE':
            if self.debiasing:
                self.loss_fct = nn.CrossEntropyLoss(reduction='none')
            else:
                self.loss_fct = nn.CrossEntropyLoss()
        elif self.loss_type.upper() == 'MSE':
            self.loss_fct = nn.MSELoss(reduction=reduction)
        elif self.loss_type.upper() == 'NLL':
            # self.loss_fct = nn.NLLLoss(reduction='none')
            # self.loss_fct = nn.BCEWithLogitsLoss()
            self.loss_fct = nn.BCELoss(reduction=reduction)
            self.m = nn.Sigmoid()
            if self.output_dim > 2:
                self.loss_fct == nn.CrossEntropyLoss(reduction=reduction)
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['CE', 'MSE', 'NLL']!")

        # parameters initialization
        self.apply(self._init_weights)
        self.log_info()

    def log_info(self):
        # print("********* Using TMF: v_u * v_i_t + b + b_i + b_u + b_T **********")
        print("********* Using TMF: v_u(t) * v_i + b + b_i + b_u + b_T **********")
        # print("********* Using TMF: v_u(t) * v_i + b_T **********")

        # print("********* Using TMF: v_u(t) * (v_i + v_T) + b_T **********")
        # self.time_embedding = nn.Embedding(self.n_periods, self.embedding_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(self.gru_layers.weight_hh_l0)
            xavier_uniform_(self.gru_layers.weight_ih_l0)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight)
            constant_(module.bias, 0.0)
        elif isinstance(module, Parameter):
            constant_(module.weight, 0.0)
    
    def _gather_indexes(self, output, gather_index):
        """Gathers the vectors at the spexific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def get_item_embedding(self, item, itemage): # [B], [B]
        idx = item * self.n_periods + itemage
        return self.item_Dyn_embedding(idx)
        # return self.item_embedding(item) + self.item_Dyn_embedding(idx)
    def get_user_embedding(self, user, itemage): # [B], [B]
        idx = user * self.n_periods + itemage
        return self.user_Dyn_embedding(idx)

    def forward(self, user, item, itemage):
        user_e = self.get_user_embedding(user, itemage) # self.user_embedding(user)
        item_e = self.item_embedding(item) # self.get_item_embedding(item, itemage)
        # # v_u(t) * v_i + b_T
        output = torch.mul(user_e, item_e).sum(-1).float() # [B, D] -> [B]
        # # v_u(t) * (v_i + v_T) + b_T
        # time_e = self.time_embedding(itemage)
        # output = torch.mul(user_e, item_e + time_e).sum(-1).float() # [B, D] -> [B]

        output += self.global_T(itemage).squeeze() + self.b + self.b_u(user).squeeze() + self.b_i(item).squeeze()
        if self.m is None:
            return output
        return self.m(output)
        
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

class TMF_variety(TMF):
    '''
    Modified time-aware matrix factorization
    r_{u,i} = p_u * q_i
    f_{u,i,t} = sigmoid(w * r_{u,i}*t + b)
    '''
    def __init__(self, config, data, debiasing=False, output_dim=2):
        super(TMF_variety, self).__init__(config, data, debiasing=False, output_dim=2)
        self.item_Dyn_embedding = None
        self.time_embedding = None
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        if self.task.upper() == 'OPPT':
            self.b_u = nn.Embedding(self.n_users, 1)
            self.b_i = nn.Embedding(self.n_items, 1)
            self.b = Parameter(torch.Tensor(1))
        self.global_T = nn.Embedding(self.n_periods, 1)
        self.apply(self._init_weights)
    
    def log_info(self):
        if self.task.upper() == 'OIPT':
            print("********* Using TMF-variety: (v_u * v_i) + b_T **********")
        else: 
            print("********* Using TMF-variety: (v_u * v_i) + b + b_i + b_u + b_T **********")
        # print("********* Using TMF-variety: (v_u * v_i) + b + b_i + b_u **********")
        
        

    def forward(self, user, item, itemage):
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        r_ui = torch.mul(user_e, item_e).sum(-1).float() # [B D] -> [B]
        # # W * p1 * T + b
        # [B 1]
        # # p1 + b
        # f_uit = torch.mul(torch.mul(r_ui, itemage), self.w) # + self.b
        if self.task.upper() == 'OIPT':
            f_uit = r_ui + self.global_T(itemage).squeeze()
        else:
            f_uit = r_ui + self.b + self.b_u(user).squeeze() + self.b_i(item).squeeze() + self.global_T(itemage).squeeze()
        
        if self.m is None:
            return f_uit
        return self.m(f_uit) # [B, D] -> [B]

class TMF_fast(TMF):
    '''
    Modified time-aware matrix factorization
    r_{u,i} = p_u * q_i
    f_{u,i,t} = sigmoid(w * r_{u,i}*t + b)
    '''
    def __init__(self, config, data, debiasing=False, output_dim=2):
        super(TMF_fast, self).__init__(config, data, debiasing=False, output_dim=2)
        print("********* Using TMF-fast: W * (v_u * v_i) * T + b **********")
        
        # define layers and loss
        self.item_Dyn_embedding = None
        self.dense = nn.Linear(1, 1)
        # # self.b = Parameter(torch.Tensor(1))
        # self.w = Parameter(torch.Tensor(1))
        self.apply(self._init_weights)

    def forward(self, user, item, itemage):
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        r_ui = torch.mul(user_e, item_e).sum(-1).float() # [B D] -> [B]
        # # W * p1 * T + b
        f_uit = self.dense(torch.mul(r_ui, itemage).unsqueeze(1)).squeeze().float() # [B 1]
        # # p1 + b
        # f_uit = torch.mul(torch.mul(r_ui, itemage), self.w) # + self.b
        f_uit = self.m(f_uit)
        return f_uit # [B, D] -> [B]

class TMF_fast_variety(TMF):
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
        self.item_Dyn_embedding = None
        self.dense = None
        # self.dense = nn.Linear(1, 1)
        # self.dense = nn.Linear(2, 1)
        # self.W1 = Parameter(torch.Tensor(1))
        # self.W2 = Parameter(torch.Tensor(1))
        # self.b = Parameter(torch.Tensor(1))

        # define layers and loss
        # self.n_factors = 2
        # if self.n_factors == 2:
        #     # print("TMF_fast_variety2: sigmoid(p1 * T1 + p2)")
        #     # print("TMF_fast_variety2': sigmoid(W1 * p1 * T1 + W2 * p2 + b)")
        #     print("...")
        # else:
        #     raise NotImplementedError("Make sure 'n_factors' in [2]!")

        self.n_factors = 1
        self.user_embedding = nn.Embedding(self.n_users * self.n_factors, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items * self.n_factors, self.embedding_size)
        # self.b_u = nn.Embedding(self.n_users, 1)
        # self.b_i = nn.Embedding(self.n_items, 1)
        # self.global_T = nn.Embedding(self.n_periods, 1)
        # self.b = Parameter(torch.Tensor(self.n_factors))

        # # self.a_i = nn.Embedding(self.n_items, 1)
        # self.a_u = nn.Embedding(self.n_users, 1)
        self.s_T = nn.Embedding(self.n_periods, 1)


        # d_p2 = 2
        # self.user_embedding2 = nn.Embedding(self.n_users, d_p2)
        # self.item_embedding2 = nn.Embedding(self.n_items, d_p2)
        # self.b2 = Parameter(torch.Tensor(1))
        # self.itemage_embedding  = nn.Embedding(self.n_periods, 1)
        # # self.itemage_embedding = nn.Embedding(self.n_periods, self.embedding_size)

        # self.item_embedding_ = nn.Embedding(self.n_items, self.embedding_size)

        # # NLL using BinaryCrossEntropy with sigmoid
        # # self.loss_fct = nn.BCEWithLogitsLoss()
        # # Then we want to try different models.
        # self.loss_fct = nn.BCELoss()
        # self.m = nn.Tanh()
        # self.m = nn.Sigmoid()
        # print("-*-*-*-* We use p_T = s_T *-*-*-*-")
        # print("-*-*-*-* We use p_T = sigmoid(s_T) *-*-*-*-")
        # print("-*-*-*-* We use p_T = sigmoid(W*T+b) *-*-*-*-")
        print("-*-*-*-* We use p_T = sigmoid(p1 * s_T) *-*-*-*-")
        # # for OPPT task
        # print("-*-*-*-* We use s_\{uit\} = p1 + (v_i * v_T) *-*-*-*-")
        # print("-*-*-*-* We use s_{uit} = p_ui + b_T + a_u * s_T *-*-*-*-")
        self.apply(self._init_weights)

    def forward(self, user, item, itemage):
        # user_e = self.user_embedding(user)
        # item_e = self.item_embedding(item)
        # output = torch.mul(user_e, item_e).sum(-1).float() # [B, D] -> [B]

        prob_scores = [] # s1, s2, ...
        for i in range(self.n_factors):
            prob_scores.append(torch.mul(self.user_embedding(self.n_users * i + user), \
                    self.item_embedding(self.n_items * i + item)).sum(-1).float())#  + self.b[i]) # [B]

        # # p1 + p2^{d=1} * T
        # p2 = torch.mul(self.user_embedding2(user), self.item_embedding2(item)).sum(-1).float() + self.b2
        # output = (torch.mul(p2, self.itemage_embedding(itemage).squeeze().float()) \
                #  + prob_scores[0]).squeeze().float() + self.global_T(itemage).squeeze()
        # # p_ui + b_T + a_u * s_T, where p_ui = v_u * v_i + b + b_u + b_i
        # p_ui = prob_scores[0] + self.b_u(user).squeeze() + self.b_i(item).squeeze()
        # output = p_ui + self.global_T(itemage).squeeze() + torch.mul(self.a_u(user), self.s_T(itemage)).squeeze()
        # output = p_ui + self.global_T(itemage).squeeze() + torch.mul(self.a_u(user).squeeze(), itemage).squeeze()
        # # p1 * T1 + p2
        # output = (torch.mul(prob_scores[1], itemage) + prob_scores[0]).squeeze().float()
        # # p1 * s_T + p2
        # output = (torch.mul(prob_scores[0], self.itemage_embedding(itemage).squeeze().float()) + prob_scores[1]).squeeze().float()
        # # p1 + v_i * v_T
        # output = prob_scores[0] + torch.mul(self.itemage_embedding(itemage), self.item_embedding_(item)).sum(-1).float()
        # # W1 * p1 * T1 + W2 * p2 + b
        # p1 = torch.mul(prob_scores[0], itemage).unsqueeze(1)
        # p2 = prob_scores[1].unsqueeze(1)
        # output = self.dense(torch.cat((p1, p2), 1)).squeeze().float()
        # output = (self.W1 * torch.mul(prob_scores[0], itemage) + self.W2 * prob_scores[1] + self.b).squeeze().float()
        # # sigmoid(P1 * s_T)
        output = torch.mul(prob_scores[0], self.s_T(itemage).squeeze().float())
        if self.m is not None:
            output = self.m(output)
        # # W * T + b
        # output = self.dense(itemage.float().unsqueeze(1)).squeeze().float()
        # # s_T 
        # output = self.s_T(itemage).squeeze()
        # output = self.m(output)
        # output = torch.clamp(output, min=0.000001, max=0.999999) # clip to make prob between 0 and 1
        return output

