import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_

class GRU4Rec(nn.Module):
    def __init__(self, config, data, debiasing=False):
        super(GRU4Rec, self).__init__()

        # load parameter info
        self.debiasing = debiasing
        self.embedding_size = int(config['embedding_size'])
        self.hidden_size = int(config['hidden_size'])
        self.loss_type = config['loss_type']
        # self.lr_decay_step = int(config['lr_decay_step'])
        self.batch_size = int(config['batch_size'])
        self.num_layers = int(config['num_layers'])
        self.n_items = data.n_items # one for padding
        self.dropout_prob = float(config['dropout_prob'])

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items+1, self.embedding_size, padding_idx=self.n_items)

        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        if self.loss_type.upper() == 'CE':
            if self.debiasing:
                self.loss_fct = nn.CrossEntropyLoss(reduction='none')
            else:
                self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['CE']!")

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

    def forward(self, item_seq, item_seq_len):
        # if (item_seq >= self.n_items).sum()>0:
        #     print(item_seq[item_seq >= self.n_items], self.n_items)
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        seq_output = self._gather_indexes(gru_output, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction['seq']
        item_seq_len = interaction['seq_len']
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction['target']
        # self.loss_type = 'CE'
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)
        if self.debiasing:
            ctr = interaction['ctr'] # [B]
            loss = torch.mul(loss, ctr).sum() # [B] -> [1]
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction['seq']
        item_seq_len = interaction['seq_len']
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores