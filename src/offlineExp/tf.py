import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_
from torch.nn.parameter import Parameter

class TF(nn.Module):
    '''
    Time-aware matrix factorization
    Cite: Collaborative filtering with temporal dynamics
    We only consider q_i(t) here when modeling r_{u,i,t}
    '''
    def __init__(self, config, data, debiasing=False):
        super(TF, self).__init__()

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

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.time_embedding = nn.Embedding(self.n_periods, self.embedding_size)
        
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
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['CE', 'MSE', 'NLL']!")

        # parameters initialization
        self.apply(self._init_weights)
        self.log_info()
    
    def log_info(self):
        # print("********* Using TF-variety: v_i * (v_u + v_t) **********")
        print("********* Using TF-variety: v_u * (v_i + v_t) **********")
        # print("********* Using TF-variety: v_u * v_i * v_t **********")
        # print("********* Using TF-variety: v_i * v_t **********")
        # self.user_embedding = None

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

    def forward(self, user, item, itemage):
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        time_e = self.time_embedding(itemage)
        # # u_v * i_v * T_v
        # ui_e = torch.mul(user_e, item_e) # [B D]
        # uit_e = torch.mul(ui_e, time_e).sum(-1).float() # [B, D] -> [B]
        # # u_v * (i_v + T_v)
        uit_e = torch.mul(user_e, time_e + item_e).sum(-1).float()
        # # i_v * (u_v + T_v)
        # uit_e = torch.mul(item_e, time_e + user_e).sum(-1).float()
        # # t_v * T_v
        # uit_e = torch.mul(item_e, time_e).sum(-1).float()
        if self.m is None:
            return uit_e
        return self.m(uit_e)
        

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

# class TMTF(TF):
#     def __init__(self, config, data, debiasing=False):
#         super(TMTF, self).__init__(config, data, debiasing)
#         # define layers and loss
#         self.user_embedding2 = nn.Embedding(self.n_users, self.embedding_size)
#         self.item_embedding2 = nn.Embedding(self.n_items, self.embedding_size)
#         self.time_embedding2 = nn.Embedding(self.n_periods, self.embedding_size)
#         self.b_T = nn.Embedding(self.n_periods, 1)
#         self.apply(self._init_weights)
    
#     def log_info(self):
#         print("********* Using TMTF: sigmoid(u1 * i1 + u2 * t1 + i2 * t2 + b_T) **********")

#     def forward(self, user, item, itemage):
#         user_e = self.user_embedding(user)
#         item_e = self.item_embedding(item)
#         time_e = self.time_embedding(itemage)
#         user_e2 = self.user_embedding2(user)
#         item_e2 = self.item_embedding2(item)
#         time_e2 = self.time_embedding2(itemage)
#         uit_e = torch.mul(user_e, item_e).sum(-1) + torch.mul(user_e2, time_e).sum(-1) + \
#             torch.mul(item_e2, time_e2).sum(-1) + self.b_T(itemage).squeeze()
#         if self.m is None:
#             return uit_e
#         return self.m(uit_e)

class TMTF(TF):
    def __init__(self, config, data, debiasing=False):
        super(TMTF, self).__init__(config, data, debiasing)
        self.b_T = nn.Embedding(self.n_periods, 1)
        self.b_u = nn.Embedding(self.n_users, 1)
        self.b_i = nn.Embedding(self.n_items, 1)
        self.b = Parameter(torch.Tensor(1))
        self.apply(self._init_weights)
    
    def log_info(self):
        # print("********* Using TMTF: v_u * (v_i + v_t) + b_T **********")
        # print("********* Using TMTF: v_u * (v_i + v_t) + v_i * v_t + b_T **********")
        if self.task.upper() == 'OIPT':
            print("********* Using TMTF: v_i * (v_u + v_t) + b_T **********")
        else:
            print("********* Using TMTF: v_u * (v_i + v_t) + b + b_i + b_u + b_T **********")

    def forward(self, user, item, itemage):
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        time_e = self.time_embedding(itemage)
        if self.task.upper() == 'OIPT':
            # # u_v * (i_v + T_v) + b_T
            uit_e = torch.mul(user_e, time_e + item_e).sum(-1).float() + self.b_T(itemage).squeeze()
        else:
            # # u_v * (i_v + T_v) + b + b_i + b_u + b_T 
            uit_e = torch.mul(user_e, time_e + item_e).sum(-1).float() + self.b + self.b_u(user).squeeze() + self.b_i(item).squeeze() + self.b_T(itemage).squeeze()
        # # u_v * i_v + u_v * T_v + i_v * T_v
        # uit_e = torch.mul(user_e, time_e + item_e).sum(-1).float() + torch.mul(item_e, time_e).sum(-1).float() + self.b_T(itemage).squeeze()
        # # v_i * (v_u + v_t) + b_T
        # uit_e = torch.mul(item_e, time_e + user_e).sum(-1).float() + self.b_T(itemage).squeeze()
        if self.m is None:
            return uit_e
        return self.m(uit_e)