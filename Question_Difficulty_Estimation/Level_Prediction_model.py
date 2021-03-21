import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


from model.Transformer_Encoder import TransformerEncoder, TransformerEncoderLayer 
from model.DuMA import DualMultiheadAttention
from model.bert.bert import BertModel

class SequenceClassification(nn.Module):

    def __init__(self, config, dropout_prob, num_labels, num_info=2, num_layers=6, num_heads=8):
        super(SequenceClassification, self).__init__()

        self.bert = BertModel(config)
        for params in self.bert.parameters():
            params.requires_grad = False

        self.embedding_dim = config.hidden_size
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_prob)

        #encoder layer
        #self attention encoder
        self_encoder_layer = TransformerEncoderLayer(d_model=self.embedding_dim, nhead=8)
        self_encoder_norm = nn.LayerNorm(self.embedding_dim)
        self_attention = TransformerEncoder(self_encoder_layer, num_layers=6, norm=self_encoder_norm)
        self.self_attention_list = nn.ModuleList([copy.deepcopy(self_attention) for _ in range(num_info)])

        #Dual Multiheadattention
        DuMA = DualMultiheadAttention(d_model=self.embedding_dim, nhead=8)
        self.dual_attention_list = nn.ModuleList([copy.deepcopy(DuMA) for _ in range(num_info-1)])

        #self.classifier = nn.Linear(self.embedding_dim*(num_info-1)*4, num_labels)
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim*(num_info-1)*4, int(self.embedding_dim)),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(int(self.embedding_dim), self.embedding_dim),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, num_labels)
        )


    def forward(self, q_vec, q_mask, q_segment, d_vec=None, d_mask=None, d_segment=None):

        # for demo system (question, utterance) fair input
        with torch.no_grad():
            q_emb = self.bert(q_vec, q_segment, q_mask)[0][-1]
            d_emb = self.bert(d_vec, d_segment, d_mask)[0][-1]

        q_emb = q_emb.permute(1, 0, 2)
        d_emb = d_emb.permute(1, 0, 2)
                
        q_rep, _ = self.self_attention_list[0](q_emb, q_emb, q_emb)
        d_rep, _ = self.self_attention_list[1](d_emb, d_emb, d_emb)

        q_d_rep, _, _ = self.dual_attention_list[0](q_rep, d_rep, d_rep)

        result_output = q_d_rep.permute(1, 2, 0)


        avg_pool = F.adaptive_avg_pool1d(result_output, 1)
        max_pool = F.adaptive_max_pool1d(result_output, 1)

        avg_pool = avg_pool.view(q_vec.size(0), -1)
        max_pool = max_pool.view(q_vec.size(0), -1)

        result = torch.cat((avg_pool, max_pool), 1)  # [batch_size, embedding_size*2]

        logits = self.classifier(result)

        return logits
