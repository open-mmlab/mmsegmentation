import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
from .utils import weight_reduce_loss


def relu_evidence(logits):
    # This function to generate evidence is used for the first example
    return F.relu(logits)


def exp_evidence(logits):
    # This one usually works better and used for the second and third examples
    # For general settings and different datasets, you may try this one first
    return torch.exp(torch.clamp(logits / 10, -10, 10))


def softplus_evidence(logits):
    # This one is another alternative and
    # usually behaves better than the relu_evidence
    return F.softplus(logits)


def mse_edl_loss(one_hot_gt, alpha, lam, num_classes):
    strength = torch.sum(alpha, dim=1, keepdim=True)
    evidence = alpha - 1
    prob = alpha / strength

    A = torch.sum((one_hot_gt - prob)**2, dim=1, keepdim=True)
    B = torch.sum(alpha * (strength - alpha) / (strength * strength * (strength + 1)), dim=1, keepdim=True)

    alpha_ = evidence * (1 - one_hot_gt) + 1
    C = lam * KL(alpha_, num_classes)

    return (A + B) + C


def KL(alpha, num_classes):
    device = alpha.device
    beta = torch.ones((1, num_classes, 1, 1), dtype=torch.float32, device=device)  # uncertain dir
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

    def __init__(self, num_classes, annealing_step=10, logit2evidence="exp", reduction="mean", loss_weight=1.0, avg_non_ignore=False, loss_name='loss_edl'):
        super(EDLLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.annealing_step = annealing_step
        self.num_classes = num_classes
        self.avg_non_ignore = avg_non_ignore
        if logit2evidence == "exp":
            self.logit2evidence = exp_evidence
        elif logit2evidence == "softplus":
            self.logit2evidence = softplus_evidence
        elif logit2evidence == "relu":
            self.logit2evidence = relu_evidence
        else:
            raise KeyError(logit2evidence)
        self.curr_epoch = 0
        self.loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        device = pred.device
        reduction = (reduction_override if reduction_override else self.reduction)
        evidence = self.logit2evidence(pred)

        alpha = evidence + 1
        target_expanded = target.data.unsqueeze(1).clone()
        mask_ignore = (target_expanded == 255)
        target_expanded[mask_ignore] = 0
        one_hot_gt = torch.zeros_like(pred, dtype=torch.uint8).scatter_(1, target_expanded, 1)

        loss = mse_edl_loss(one_hot_gt=one_hot_gt, alpha=alpha, lam=self.lam, num_classes=self.num_classes)

        if ignore_index:
            loss = torch.where(mask_ignore, torch.zeros_like(loss), loss)
        if (avg_factor is None) and self.avg_non_ignore and reduction == 'mean':
            avg_factor = target.numel() - (target == ignore_index).sum().item()
        if weight is not None:
            weight = weight.float()
        loss_cls = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
        return self.loss_weight * loss_cls

    def get_evidence(self,
                     pred,
                     target,
                     ignore_index=255):
        pred_detached = pred.detach()
        evidence = self.logit2evidence(pred_detached)
        pred_cls = torch.argmax(pred.data.clone(), dim=1, keepdim=True)
        gt_cls = target.data.unsqueeze(1).clone()
        mask_ignore = (gt_cls == ignore_index)
        succ = torch.logical_and((pred_cls == gt_cls), ~mask_ignore)
        fail = torch.logical_and(~(pred_cls == gt_cls), ~mask_ignore)
        mean_evidence = evidence.sum(dim=1, keepdim=True).mean()
        mean_fail_evidence = (evidence.sum(dim=1, keepdim=True) * fail).sum() / (fail.sum() + 1e-20)
        mean_succ_evidence = (evidence.sum(dim=1, keepdim=True) * succ).sum() / (succ.sum() + 1e-20)
        return mean_evidence, mean_succ_evidence, mean_fail_evidence

    @property
    def lam(self):
        return torch.min(torch.tensor(1.0, dtype=torch.float32), torch.tensor(self.curr_epoch / self.annealing_step, dtype=torch.float32))

    def get_probs(self, pred):
        pred_detached = pred.detach()
        evidence = self.logit2evidence(pred_detached)
        alpha = evidence + 1
        strength = alpha.sum(dim=1, keepdim=True)
        u = self.num_classes / strength
        prob = alpha / strength
        return prob, u
