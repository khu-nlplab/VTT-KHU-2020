import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from metric.stat_metric import StatMetric

from . import loss_utils

class ConfidentOracleLoss(_Loss):
    def __init__(self, beta, num_labels,num_models , num_assign, padding_idx):
        super(ConfidentOracleLoss, self).__init__()

        # self.eps = eps
        self.beta =beta
        self.num_labels = num_labels
        self.padding_idx = padding_idx
        self.num_models = num_models
        self.num_assign = num_assign

        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=padding_idx, reduction='none')
        # self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=padding_idx)
        # super(CrossEntropyLoss, self).__init__(ignore_index=padding_idx)

    @staticmethod
    def get_metric():
        return {'oracle_loss': StatMetric(output_transform=lambda x: (x[1]['oracle_loss'], x[2]))}

    def forward(self, hypos, tgt):

        """

        Args:
            hypo: ([B, M, C])
            tgt: [B,]
        Returns:

        """

        B = tgt.size(0)
        #print(hypos.size())
        hypos = torch.unbind(hypos, dim=1)
        # hypos = hypos.contiguous()
        # tgt = tgt.contiguous()


        task_loss_list = [self.cross_entropy_loss.forward(hypo.view(-1, hypo.shape[-1]), tgt.view(-1)) for hypo in hypos]
        task_loss_tensor = torch.stack(task_loss_list, dim=0)
        # if hypo.nelement() == 0 or tgt.nelement() == 0:  # failsafe for empty tensor
        #     loss = None



        # loss = super().forward(hypo.view(-1, hypo.shape[-1]),
        #                     tgt.view(-1))
        prob_list = [F.softmax(hypo, dim=1) for hypo in hypos]
        # nonspecialist_loss_list = [self.beta * F.kl_div(torch.Tensor([1 / self.num_labels] * self.num_labels), prob, None, None, 'mean') for prob in prob_list]
        nonspecialist_loss_list = [self.beta \
                                   * (-prob.log().mean(dim=1)).add(-np.log(self.num_labels)) \
                                   for prob in prob_list]

        # print(task_loss_list)
        # print(nonspecialist_loss_list)
        nonspecialist_loss_tensor = torch.stack(nonspecialist_loss_list, dim=0)

        assign_idx, not_assign_idx = loss_utils.get_combinations(self.num_models , self.num_assign)

        # print(assign_idx)
        # print(not_assign_idx)

        oracle_loss_list = []
        num_combinations = len(assign_idx)
        #print(num_combinations)
        for c in range(num_combinations):
            specialized = [task_loss_list[idx] for idx in assign_idx[c]]
            non_specialized = [nonspecialist_loss_list[idx] for idx in not_assign_idx[c]]

            oracle_loss_list.append(sum(specialized) + sum(non_specialized))

        oracle_loss_tensor = torch.stack(oracle_loss_list, dim=0) # [nc, B]
        # print(oracle_loss_list)
        # print(oracle_loss_tensor.size())
        # print(oracle_loss_tensor)
        # exit()
        min_val , min_idx = oracle_loss_tensor.t().min(dim=1) # [B,]

        # CMCL_v1
        # sampling labels stochastically
        np_random_labels = np.random.randint(0, self.num_labels, size=(self.num_models, B))
        random_labels = torch.autograd.Variable(
            torch.from_numpy(np_random_labels), requires_grad=False).long()  # [m, B]
        one_mask = torch.autograd.Variable(
            torch.from_numpy(np.ones(B)).float(), requires_grad=False)  # [B]
        beta_mask = torch.autograd.Variable(
            torch.from_numpy(np.full((B), self.beta)).float(), requires_grad=False)  # [B]

        if torch.cuda.is_available():
            random_labels = random_labels.cuda()
            one_mask = one_mask.cuda()
            beta_mask = beta_mask.cuda()

            # compute loss
        max_k = min_idx.size()[0]
        for mi in range(self.num_models):
            for topk in range(max_k):
                null_mask = min_idx[topk].eq(-1).long()  # [B]
                if null_mask.sum().item() == B:
                    continue
                selected_mask = min_idx[topk].eq(mi).long()  # [B]
                if torch.cuda.is_available():
                    selected_mask = selected_mask.cuda()

                if topk == 0:
                    sampled_labels = selected_mask * tgt + ((1-selected_mask) * random_labels[mi])
                    finally_selected = selected_mask
                else:
                    sampled_labels = selected_mask * tgt + ((1-selected_mask) * sampled_labels)
                    finally_selected += selected_mask

            finally_selected = finally_selected.ge(1.0).float()
            coeff = finally_selected * one_mask + ((1 - finally_selected) * beta_mask)
            loss = coeff * F.cross_entropy(hypos[mi], sampled_labels, reduce=False)
            if mi == 0:
                total_loss = loss.sum() / self.num_assign / B
            else:
                total_loss += (loss.sum() / self.num_assign / B)

        return total_loss, {'oracle_loss': total_loss.item()}

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
        beta = args.get("beta", 75.0)
        num_labels = args.get("num_labels", 5)
        num_models = args.get("num_models", 2)
        num_assign = args.get("num_assign", 1)
        padding_idx = vocab.stoi[vocab.pad]

        return cls(beta=beta, num_labels=num_labels,num_models = num_models, num_assign=num_assign, padding_idx=padding_idx)