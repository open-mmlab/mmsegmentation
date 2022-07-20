import torch
import torch.nn as nn

from ..builder import LOSSES


@LOSSES.register_module
class BeliefMatchingLoss(nn.Module):
    def __init__(self, coef, prior=1., loss_weight=1., reduction="mean", loss_name="bm_loss"):
        super(BeliefMatchingLoss, self).__init__()
        self.prior = prior
        self.coeff = coef
        self.loss_weight = loss_weight
        self.loss_name = loss_name
        self.reduction = reduction

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255):
        '''
        Compute loss: kl - evi

        '''
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        alphas = torch.exp(pred)
        betas = torch.ones_like(pred) * self.prior

        # compute log-likelihood loss: psi(alpha_target) - psi(alpha_zero)

        target_ = target.data.clone()
        mask_ignore = (target_ == 255)
        target_[mask_ignore] = 0

        a_ans = torch.gather(alphas, 1, target_.unsqueeze(1)).squeeze(1)
        a_zero = torch.sum(alphas, 1)
        ll_loss = torch.digamma(a_ans) - torch.digamma(a_zero)

        # compute kl loss: loss1 + loss2
        #       loss1 = log_gamma(alpha_zero) - \sum_k log_gamma(alpha_zero)
        #       loss2 = sum_k (alpha_k - beta_k) (digamma(alpha_k) - digamma(alpha_zero) )

        loss1 = torch.lgamma(a_zero) - torch.sum(torch.lgamma(alphas), 1)

        loss2 = torch.sum((alphas - betas) * (torch.digamma(alphas) - torch.digamma(a_zero.unsqueeze(1))), 1)
        kl_loss = loss1 + loss2

        loss = ((self.coeff * kl_loss - ll_loss))
        if ignore_index:
            loss = torch.where(mask_ignore, torch.zeros_like(loss), loss)

        avg_factor = target.numel() - (target == ignore_index).sum().item()
        if reduction == 'mean':
            loss_cls = loss.sum() / avg_factor
        elif reduction == 'sum':
            loss_cls = loss.sum()
        else:
            loss_cls = loss

        return loss_cls
