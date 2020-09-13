import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from metric.stat_metric import StatMetric

from . import loss_utils

class KnowledgeDistillationOracleLoss(_Loss):
    def __init__(self, beta, tau, num_labels, num_models , num_assign, padding_idx,  kd_method="avg"):
        super(KnowledgeDistillationOracleLoss, self).__init__()

        # self.eps = eps
        self.beta = beta
        self.tau = tau
        self.num_labels = num_labels
        self.padding_idx = padding_idx
        self.num_models = num_models
        self.num_assign = num_assign
        self.kd_method = kd_method

        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=padding_idx, reduction='none')
        # self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=padding_idx)
        # super(CrossEntropyLoss, self).__init__(ignore_index=padding_idx)

    @staticmethod
    def get_metric():
        return {'oracle_loss': StatMetric(output_transform=lambda x: (x[1]['knowledge_distillation_oracle_loss'], x[2]))}

    def forward(self, hypos, tgt):

        """

        Args:
            hypos: [([B, M, C]), ([B, M, C])]
            tgt: [B,]
        Returns:

        """

        B = tgt.size(0)

        student_logits = hypos[0]
        student_logits = torch.unbind(student_logits, dim=1) # M * ([B, C])
        teacher_logits = hypos[-1]
        teacher_logits = torch.unbind(teacher_logits, dim=1) # M * ([B, C])

        student_task_loss_list = [self.cross_entropy_loss.forward(logit.view(-1, logit.shape[-1]), tgt.view(-1)) for logit in student_logits]
        student_task_loss_tensor = torch.stack(student_task_loss_list, dim=0)


        # loss = super().forward(hypo.view(-1, hypo.shape[-1]),
        #                     tgt.view(-1))

        if self.kd_method == "avg":
            nonspecialist_loss_list = [self.beta * loss for loss in self.compute_akld(student_logits, teacher_logits, self.tau)]
        elif self.kd_method == "none":
            nonspecialist_loss_list = [self.beta * loss for loss in self.compute_kld(student_logits, teacher_logits, self.tau)]

        nonspecialist_loss_tensor = torch.stack(nonspecialist_loss_list, dim=0)

        assign_idx, not_assign_idx = loss_utils.get_combinations(self.num_models, self.num_assign)


        oracle_loss_list = []
        num_combinations = len(assign_idx)
        for c in range(num_combinations):
            specialized = [student_task_loss_list[idx] for idx in assign_idx[c]]
            non_specialized = [nonspecialist_loss_list[idx] for idx in not_assign_idx[c]]

            oracle_loss_list.append(sum(specialized) + sum(non_specialized))

        oracle_loss_tensor = torch.stack(oracle_loss_list, dim=0) # [nc, B]

        min_val, min_idx = oracle_loss_tensor.t().min(dim=1) # [B,]

        total_loss = min_val.sum() / B

        return total_loss, {'knowledge_distillation_oracle_loss': total_loss.item()}

    def _reduce(self, t):
        func = {
            'none': lambda x: x,
            'mean': lambda x: x.mean(),
            'sum': lambda x: x.sum()
        }[self.reduction]

        return func(t)

    def compute_kld(self, inputs, targets, tau=1.0):

        kld_losses = []
        """
        teacher = torch.stack(targets, dim=1).mean(dim=1) / tau
        soft_teacher = F.softmax(teacher, dim=1)

        for student in inputs:
            student = student / tau

            logP = F.log_softmax(student, dim=1)

            kld_loss = -(soft_teacher * logP).mean(dim=1)
            kld_losses.append(kld_loss)

        """
        for student, teacher in zip(inputs, targets):

            student = student / tau
            teacher = teacher / tau
            
            logP = F.log_softmax(student, dim=1)
            soft_teacher = F.softmax(teacher, dim=1)

            kld_loss = -(soft_teacher * logP).mean(dim=1)
            kld_losses.append(kld_loss)


        return kld_losses


    def compute_akld(self, inputs, targets, tau=1.0):
        akld_losses = []

        teacher = torch.mean(torch.stack(targets, dim=1), dim=1) # [B, ]

        for student, teacher in zip(inputs, targets):
            student = student / tau
            teacher = teacher / tau

            logP = F.log_softmax(student, dim=1)
            soft_teacher = F.softmax(teacher, dim=1)

            kld_loss = -(soft_teacher * logP).mean(dim=1)
            akld_losses.append(kld_loss)

        return akld_losses


    """
    def compute_mkld(self, inputs, targets, tau=1.0):

        mkld_losses = []
        for student in inputs:
            multi_teachers_loss = []
            for teacher in targets:
                mean_squared_error = ((student-teacher)**2).mean(dim=1) # [B,]
                multi_teachers_loss.append(mean_squared_error)

            loss_sum = torch.sum(torch.stack(multi_teachers_loss, dim=1), dim=1) # [B, n_teachers] -> [B,]
            norm_loss_sum = loss_sum / len(targets) # [B,]
            mkld_losses.append(norm_loss_sum) # [B, n_students]

        return mkld_losses
    """
    @classmethod
    def resolve_args(cls, args, vocab):
        # eps = args.get("label_smoothing", 0)
        beta = args.get("beta", 0.75)
        tau = args.get("tau", 1.0)
        num_labels = args.get("num_labels", 5)
        num_models = args.get("num_models", 2)
        num_assign = args.get("num_assign", 1)
        kd_method = args.get("kd_method", "avg")
        padding_idx = vocab.stoi[vocab.pad]

        return cls(beta=beta, tau=tau, num_labels=num_labels,num_models = num_models,
                   num_assign=num_assign, padding_idx=padding_idx, kd_method="avg")