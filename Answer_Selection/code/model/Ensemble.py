import torch
from torch import nn
from torch.nn import functional as F

from . import get_model
from . rnn import RNNEncoder, max_along_time

class Ensemble(nn.Module):

    def __init__(self, args, vocab, input_size, num_choice=5):
        super().__init__()


        self.args = args
        self.input_size = input_size
        self.output_size = num_choice

        self.vocab = vocab
        V = len(vocab)
        D = input_size

        self.ckpts = self._get_model_ckpts()
        self.n_experts = len(self.ckpts)
        self.experts = self._load_models()

        self.teachers=None
        if self.args.mode == "train" and self.args.kd is True:
            self.teachers = self._load_models()
            self.teachers.eval()


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
        # print('Load pretrained embedding ...')
        # self.embedding.weight.data.copy_(pretrained_embedding)
        #self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        pass

    def _load_models(self):
        moduleList = nn.ModuleList()

        if "experts" in self.__dict__.keys():
            print("models are already existed")
            return self.experts

        print("Loading pretrained models")
        for name, dt in self.ckpts:
            model = get_model(dt['args'], dt['vocab'])
            model.load_embedding(dt['vocab'])
            model.load_state_dict(dt['model'])
            moduleList.append(model)

        return moduleList

    def forward(self, que, answers, **features):

        e_outputs = []
        for expert in self.experts:
            e_output = expert(que, answers, **features) # [batch , answers]
            # soft_output = F.softmax(output, dim=1) # [batch , answers]
            e_outputs.append(e_output)

        e_outputs = torch.stack(e_outputs, dim=1)  # [batch , n_experts, answers]
        outputs = e_outputs

        if self.teachers is not None:
            t_outputs = []
            with torch.no_grad():
                for teacher in self.teachers:
                    t_output = teacher(que, answers, **features)
                    t_outputs.append(t_output)

            t_outputs = torch.stack(t_outputs, dim=1) # [batch, n_teachers, answers]
            outputs = (outputs, t_outputs,)

        return outputs







