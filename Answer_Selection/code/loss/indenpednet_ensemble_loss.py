import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from metric.stat_metric import StatMetric

from . import loss_utils

class IndependentEnsembleLoss(_Loss):
    def __init__(self, beta, num_labels,num_models , num_assign, padding_idx):
        super(IndependentEnsembleLoss, self).__init__()

        # self.eps = eps
        self.beta =beta
        self.num_labels = num_labels
        self.padding_idx = padding_idx
        self.num_models = num_models
        self.num_assign = num_assign

        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=padding_idx, reduction='none')

    @staticmethod
    def get_metric():
        return {'ensemble_loss': StatMetric(output_transform=lambda x: (x[1]['ensemble_loss'], x[2]))}

    def forward(self, hypos, tgt):

        """

        Args:
            hypo: ([B, M, C])
            tgt: [B,]
        Returns:

        """

        B = tgt.size(0)
        hypos = torch.unbind(hypos, dim=1)


        total_loss = [self.cross_entropy_loss.forward(hypo.view(-1, hypo.shape[-1]), tgt.view(-1)) for hypo in hypos]

        total_loss = torch.stack(total_loss, dim=1)
        total_loss = torch.sum(total_loss, dim=1)
        total_loss = total_loss.mean()

        return total_loss, {'ensemble_loss': total_loss.item()}

    def _reduce(self, t):
        func = {
            'none': lambda x: x,
            'mean': lambda x: x.mean(),
            'sum': lambda x: x.sum()
        }[self.reduction]

        return func(t)

    @classmethod
    def resolve_args(cls, args, vocab):
        # eps = args.get("label_smoothing", 0)
        beta = args.get("beta", 0.75)
        num_labels = args.get("num_labels", 5)
        num_models = args.get("num_models", 2)
        num_assign = args.get("num_assign", 1)
        padding_idx = vocab.stoi[vocab.pad]

        return cls(beta=beta, num_labels=num_labels,num_models = num_models, num_assign=num_assign, padding_idx=padding_idx)