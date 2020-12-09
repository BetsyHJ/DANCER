import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_

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
        self.n_months = data.n_months 
        self.n_users = data.n_users

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.item_Dyn_embedding = nn.Embedding(self.n_items * self.n_months, self.embedding_size)
        # self.item_Dyn_embedding = [] # [T N D]
        # for t in range(self.n_months):
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
        idx = item * self.n_months + itemage
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
                self.item_Dyn_embedding.weight.view(self.n_items, self.n_months, self.embedding_size) # [N D] + [N T D] -> [N T D]
        test_items_emb = test_items_emb.view(self.n_items * self.n_months, self.embedding_size) #[N*T D]
        scores = torch.matmul(self.user_embedding(user), test_items_emb.transpose(0, 1))  # [B D], [D N*T] -> [B N*T]
        return scores.view(-1, self.n_items, self.n_months) # [B N T]