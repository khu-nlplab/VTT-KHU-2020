import torch
from torch import nn
from torch.nn import functional as F

from . import get_model
from . rnn import RNNEncoder, max_along_time

class MOE(nn.Module):

    def __init__(self, args, vocab, input_size, num_choice=5):
        super().__init__()

        self.args = args
        self.input_size = input_size
        self.output_size = num_choice

        self.vocab = vocab
        V = len(vocab)
        D = input_size

        self.embedding = nn.Embedding(V, D)

        self.lstm_raw = RNNEncoder(D, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")

        self.ckpts = self._get_model_ckpts()
        self.n_experts = len(self.ckpts)
        self.experts = self._load_models()

        self.w_gate = nn.Parameter(torch.zeros(300, self.n_experts), requires_grad=True)



    @classmethod
    def resolve_args(cls, args, vocab):
        return cls(args, vocab, args.n_dim)

    def _get_model_ckpts(self):
        models = []
        ckpt_paths = sorted(self.args.ckpt_path.glob('*.pickle'), reverse=False)
        assert len(ckpt_paths) > 0, "no ckpt candidate for {}".format(self.args.ckpt_path)

        for ckpt_path in ckpt_paths:
            print("loading from {}".format(ckpt_path))
            dt = torch.load(ckpt_path)
            models.append((ckpt_path.stem, dt))

        return models

    def load_embedding(self, pretrained_embedding):
        print('Load pretrained embedding ...')
        # self.embedding.weight.data.copy_(pretrained_embedding)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))

    def _load_models(self):
        moduleList = nn.ModuleList()

        for name, dt in self.ckpts:

            model = get_model(dt['args'], dt['vocab'])
            model.load_embedding(dt['vocab'])
            model.load_state_dict(dt['model'])
            moduleList.append(model)

        return moduleList

    def forward(self, que, answers, **features):

        e_q = self.embedding(que)
        q_len = features['que_len']
        e_q, _ = self.lstm_raw(e_q, q_len) # [batch, len ,emb_dim]

        max_out = max_along_time(e_q, q_len) # [batch,  emb_dim]
        importance = F.softmax(torch.matmul(max_out, self.w_gate), dim=1) # [batch,  n_expert]

        outputs = []
        for expert in self.experts:
            expert.eval()
            with torch.no_grad():
                output = expert(que, answers, **features) # [batch , answers]
                soft_output = F.softmax(output, dim=1) # [batch , outputs]
                outputs.append(soft_output)

        outputs = torch.stack(outputs, dim=1) # [batch , n_experts, answers]
        importance = importance.unsqueeze(dim=2).repeat([1, 1, self.output_size]) # [batch, n_experts, answers]


        weighted_outputs = outputs * importance # [batch, n_experts, answers]

        combined_outputs = torch.sum(weighted_outputs, dim=1) # [batch, answers]

        return combined_outputs







