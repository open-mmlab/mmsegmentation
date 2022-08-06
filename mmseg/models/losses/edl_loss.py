import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES


def relu_evidence(logits):
    # This function to generate evidence is used for the first example
    return F.relu(logits)


def exp_evidence(logits):
    # This one usually works better and used for the second and third examples
    # For general settings and different datasets, you may try this one first
    return torch.exp(torch.clamp(logits, -10, 10))


def softplus_evidence(logits):
    # This one is another alternative and
    # usually behaves better than the relu_evidence
    return F.softplus(logits)


def loglikelihood_loss(y, alpha):
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    return loglikelihood_err, loglikelihood_var


def mse_edl_loss(one_hot_gt, alpha, num_classes):
    strength = torch.sum(alpha, dim=1, keepdim=True)
    prob = alpha / strength
    # L_err
    A = torch.sum((one_hot_gt - prob)**2, dim=1, keepdim=True)
    # L_var
    B = torch.sum(alpha * (strength - alpha) / (strength * (strength + 1)), dim=1, keepdim=True)
    # L_KL
    alpha_kl = (alpha - 1) * (1 - one_hot_gt) + 1
    C = KL(alpha_kl, num_classes)

    return A, B, C


def ce_edl_loss(one_hot_gt, alpha, num_classes, func):
    strength = torch.sum(alpha, dim=1, keepdim=True)
    # L_err
    A = torch.sum(one_hot_gt * (func(strength) - func(alpha)), axis=1, keepdims=True)
    # L_kl
    alpha_kl = (alpha - 1) * (1 - one_hot_gt) + 1
    C = KL(alpha_kl, num_classes)
    return A, C


def KL(alpha, num_classes):
    beta = torch.ones((1, num_classes, 1, 1), dtype=torch.float32, device=alpha.device)  # uncertain dir
    strength_alpha = torch.sum(alpha, dim=1, keepdim=True)
    strength_beta = torch.sum(beta, dim=1, keepdim=True)

    lnB = torch.lgamma(strength_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(strength_beta)

    dg0 = torch.digamma(strength_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


@ LOSSES.register_module
class EDLLoss(nn.Module):

    def __init__(self, num_classes, loss_variant="mse", annealing_step=10, annealing_method="step", logit2evidence="exp", reduction="mean",
                 loss_weight=1.0, avg_non_ignore=True, loss_name='loss_edl'):
        super(EDLLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.annealing_step = annealing_step
        self.num_classes = num_classes
        self.avg_non_ignore = avg_non_ignore
        self.annealing_start = 0.01
        self.annealing_method = annealing_method
        if logit2evidence == "exp":
            self.logit2evidence = exp_evidence
        elif logit2evidence == "softplus":
            self.logit2evidence = softplus_evidence
        elif logit2evidence == "relu":
            self.logit2evidence = relu_evidence
        else:
            raise KeyError(logit2evidence)
        self.epoch_num = 0
        self.total_epochs = 70
        self.loss_name = "_".join([loss_name, loss_variant])
        self.last_A = 0
        self.last_B = 0
        self.last_C = 0

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        evidence = self.logit2evidence(pred)

        alpha = evidence + 1
        target_expanded = target.data.unsqueeze(1).clone()
        mask_ignore = (target_expanded == 255)
        target_expanded[mask_ignore] = 0
        one_hot_gt = torch.zeros_like(pred, dtype=torch.uint8).scatter_(1, target_expanded, 1)

        if self.loss_name.endswith("mse"):  # Eq. 5 MSE
            A, B, C = mse_edl_loss(one_hot_gt, alpha, self.num_classes)
            loss = A + B + self.lam * C
        elif self.loss_name.endswith("ce"):  # Eq. 4 CrossEntropy
            A, C = ce_edl_loss(one_hot_gt, alpha, self.num_classes, func=torch.digamma)
            loss = A + self.lam * C
        elif self.loss_name.endswith("mll"):  # Eq. 3 Maximum Likelihood Type II
            A, C = ce_edl_loss(one_hot_gt, alpha, self.num_classes, func=torch.log)
            loss = A + self.lam * C
        else:
            raise NotImplementedError

        if ignore_index:
            loss = torch.where(mask_ignore, torch.zeros_like(loss), loss)

        avg_factor = target.numel() - (target == ignore_index).sum().item()
        if reduction == 'mean':
            loss_cls = loss.sum() / avg_factor
        elif reduction == 'sum':
            loss_cls = loss.sum()
        else:
            loss_cls = loss

        self.last_A = A.detach()
        self.last_A = torch.where(mask_ignore, torch.zeros_like(self.last_A), self.last_A).sum() / avg_factor
        self.last_C = C.detach()
        self.last_C = torch.where(mask_ignore, torch.zeros_like(self.last_C), self.last_C).sum() / avg_factor
        if self.loss_name.endswith("mse"):
            self.last_B = B.detach()
            self.last_B = torch.where(mask_ignore, torch.zeros_like(self.last_B), self.last_B).sum() / avg_factor

        return self.loss_weight * loss_cls

    def get_logs(self,
                 pred,
                 target,
                 ignore_index=255):
        logs = {}
        pred_detached = pred.detach()
        evidence = self.logit2evidence(pred_detached)
        alpha = evidence + 1
        strength = alpha.sum(dim=1, keepdim=True)
        u = self.num_classes / strength
        prob = alpha / strength
        max_prob, pred_cls = torch.max(prob.data.clone(), dim=1, keepdim=True)
        gt_cls = target.data.unsqueeze(1).clone()
        mask_ignore = (gt_cls == ignore_index)
        succ = torch.logical_and((pred_cls == gt_cls), ~mask_ignore)
        fail = torch.logical_and(~(pred_cls == gt_cls), ~mask_ignore)
        logs["mean_evidence"] = evidence.sum(dim=1, keepdim=True).mean()
        logs["mean_fail_evidence"] = (evidence.sum(dim=1, keepdim=True) * fail).sum() / (fail.sum() + 1e-20)
        logs["mean_succ_evidence"] = (evidence.sum(dim=1, keepdim=True) * succ).sum() / (succ.sum() + 1e-20)
        logs["mean_L_err"] = self.last_A
        if self.loss_name.endswith("mse"):
            logs["mean_L_var"] = self.last_B
        logs["mean_L_kl"] = self.last_C
        logs["lam"] = self.lam
        logs["avg_max_prob"] = (max_prob * ~mask_ignore).sum() / ((~mask_ignore).sum() + 1e-20)
        logs["avg_uncertainty"] = (u * ~mask_ignore).sum() / ((~mask_ignore).sum() + 1e-20)
        logs["acc_seg"] = succ.sum() / (fail.sum() + succ.sum()) * 100

        return logs

    @ property
    def lam(self):
        # annealing coefficient
        if self.annealing_method == 'step':
            annealing_coef = torch.min(torch.tensor(1.0, dtype=torch.float32), torch.tensor(
                self.epoch_num / self.annealing_step, dtype=torch.float32))
        elif self.annealing_method == 'exp':
            annealing_start = torch.tensor(self.annealing_start, dtype=torch.float32)
            annealing_coef = annealing_start * torch.exp(-torch.log(annealing_start) / self.total_epoch * self.epoch_num)
        else:
            raise NotImplementedError
        return annealing_coef

    def get_probs(self, pred):
        pred_detached = pred.detach()
        evidence = self.logit2evidence(pred_detached)
        alpha = evidence + 1
        strength = alpha.sum(dim=1, keepdim=True)
        u = self.num_classes / strength
        prob = alpha / strength
        return prob, u
