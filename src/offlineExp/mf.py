import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_
from torch.nn.parameter import Parameter

class MF(nn.Module):
    '''
    Time-aware matrix factorization
    Cite: Collaborative filtering with temporal dynamics
    We only consider q_i(t) here when modeling r_{u,i,t}
    '''
    def __init__(self, config, data, debiasing=False, output_dim=2):
        super(MF, self).__init__()

        self.task = config['task']
        # load parameter info
        self.debiasing = debiasing
        self.embedding_size = int(config['embedding_size'])
        self.loss_type = config['loss_type']
        # self.lr_decay_step = int(config['lr_decay_step'])
        self.batch_size = int(config['batch_size'])
        self.n_items = data.n_items 
        self.n_users = data.n_users
        self.output_dim = output_dim

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        
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
                self.m = nn.Softmax()
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

    def forward(self, user, item):
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        # if self.loss_type.upper() == 'NLL':
        #     scores = torch.mul(user_e, item_e).sum(-1).float() # [B, D] -> [B]
        #     scores = torch.sigmoid(scores).unsqueeze(-1) #[B 1] for obser
        #     return torch.cat((1.0 - scores, scores), -1) # [B, 2]
        output = torch.mul(user_e, item_e).sum(-1).float() # [B, D] -> [B]
        if self.m is None:
            return output
        return self.m(output)
        
        
    def calculate_loss(self, interaction):
        user = interaction['user']
        item = interaction['item']
        pred = self.forward(user, item)
        target = interaction['target'].float()
        loss = self.loss_fct(pred, target)
        # if self.debiasing:
        #     ctr = torch.reciprocal(interaction['ctr']) # [B]
        #     loss = torch.mul(loss, ctr).sum() # [B] -> [1]
        return loss

    def predict(self, interaction):
        user = interaction['user']
        item = interaction['item']
        pred = self.forward(user, item)
        return pred

    def full_sort_predict(self, interaction):
        user = interaction['user']
        test_items_emb = self.item_embedding.weight.view(self.n_items, 1, self.embedding_size) # [N D]
        scores = torch.matmul(self.user_embedding(user), test_items_emb.transpose(0, 1))  # [B D], [D N] -> [B N]
        return scores

class MF_dnn(MF):
    def __init__(self, config, data, debiasing=False):
        super(MF_dnn, self).__init__(config, data, debiasing)
        # self.dense = nn.Linear(1, 1)
        self.b = Parameter(torch.Tensor(1))
        # self.w = Parameter(torch.Tensor(1))


class MF_v(MF):
    def __init__(self, config, data, debiasing=False):
        super(MF_v, self).__init__(config, data, debiasing)
        # self.dense = nn.Linear(1, 1)
        self.b = Parameter(torch.Tensor(1))
        self.b_u = nn.Embedding(self.n_users, 1)
        self.b_i = nn.Embedding(self.n_items, 1)
        # self.w = Parameter(torch.Tensor(1))
        self.apply(self._init_weights)
        print('-*-*-*-* We use s_{uit} = v_u * v_i + b + b_u + b_i *-*-*-*-')

    def forward(self, user, item):
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        r_ui = torch.mul(user_e, item_e).sum(-1).float() # [B, D] -> [B]
        # # W * v_u * v_i + b
        # f_uit = self.dense(r_ui.unsqueeze(1)).squeeze().float() # [B]
        # # v_u * v_i + b
        # f_uit = r_ui + self.b
        # # v_u * v_i + b_u + b_i + b
        f_uit = r_ui + self.b + self.b_u(user).squeeze() + self.b_i(item).squeeze()
        return f_uit