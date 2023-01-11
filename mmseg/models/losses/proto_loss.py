# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F



class FSCELoss(nn.Module):
    def __init__(self, configer=None):
        super(FSCELoss, self).__init__()
        weight = None
        weight = [0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
                      1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                      1.0865, 1.0955, 1.0865, 1.1529, 1.0507]
        weight = torch.FloatTensor(weight).cuda()

        reduction = 'elementwise_mean'

        ignore_index = -1


        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, inputs, *targets, weights=None, **kwargs):
        loss = 0.0
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)

            for i in range(len(inputs)):
                if len(targets) > 1:
                    target = self._scale_target(targets[i], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)
                else:
                    target = self._scale_target(targets[0], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)

        else:
            target = self._scale_target(targets[0], (inputs.size(2), inputs.size(3)))
            loss = self.ce_loss(inputs, target)

        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()

class FSAuxCELoss(nn.Module):
    def __init__(self, configer=None):
        super(FSAuxCELoss, self).__init__()
        self.ce_loss = FSCELoss(self.configer)

    def forward(self, inputs, targets, **kwargs):
        aux_out, seg_out = inputs
        seg_loss = self.ce_loss(seg_out, targets)
        aux_loss = self.ce_loss(aux_out, targets)
        loss = 1.0 * seg_loss
        loss = loss + 0.4 * aux_loss
        return loss

class MSFSAuxRMILoss(nn.Module):
    def __init__(self, configer=None):
        super(MSFSAuxRMILoss, self).__init__()
        self.ce_loss = FSCELoss(self.configer)
        self.rmi_loss = RMILoss(self.configer)

    def forward(self, inputs, targets, **kwargs):
        aux_out = inputs['aux']
        seg_out = inputs['pred']
        pred_05x = inputs['pred_05x']
        pred_10x = inputs['pred_10x']

        aux_loss = self.ce_loss(aux_out, targets)
        seg_loss = self.rmi_loss(seg_out, targets)
        loss = 1.0 * seg_loss
        loss = loss + 0.4 * aux_loss

        scaled_pred_05x = torch.nn.functional.interpolate(pred_05x, size=(seg_out.size(2), seg_out.size(3)),
                                                          mode='bilinear', align_corners=False)
        loss_lo = self.ce_loss(scaled_pred_05x, targets)
        loss_hi = self.ce_loss(pred_10x, targets)
        loss += 0.05 * loss_lo
        loss += 0.05 * loss_hi

        return loss

class PPC(nn.Module, ABC):
    def __init__(self, configer):
        super(PPC, self).__init__()

        self.configer = configer

        self.ignore_label = -1


    def forward(self, contrast_logits, contrast_target):
        loss_ppc = F.cross_entropy(contrast_logits, contrast_target.long(), ignore_index=self.ignore_label)

        return loss_ppc


class PPD(nn.Module, ABC):
    def __init__(self, configer):
        super(PPD, self).__init__()

        self.configer = configer

        self.ignore_label = -1

    def forward(self, contrast_logits, contrast_target):
        contrast_logits = contrast_logits[contrast_target != self.ignore_label, :]
        contrast_target = contrast_target[contrast_target != self.ignore_label]

        logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
        loss_ppd = (1 - logits).pow(2).mean()

        return loss_ppd


class PixelPrototypeCELoss(nn.Module, ABC):
    def __init__(self, configer=None):
        super(PixelPrototypeCELoss, self).__init__()

        ignore_index = -1

        self.loss_ppc_weight = 0.01
        self.loss_ppd_weight = 0.001

        self.use_rmi = False
        if self.use_rmi:
            self.seg_criterion = FSAuxRMILoss(configer=configer)
        else:
            self.seg_criterion = FSCELoss(configer=configer)

        self.ppc_criterion = PPC(configer=configer)
        self.ppd_criterion = PPD(configer=configer)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if isinstance(preds, dict):
            assert "seg" in preds
            assert "logits" in preds
            assert "target" in preds

            seg = preds['seg']
            contrast_logits = preds['logits']
            contrast_target = preds['target']
            loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
            loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)

            pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
            loss = self.seg_criterion(pred, target)
            return loss + self.loss_ppc_weight * loss_ppc + self.loss_ppd_weight * loss_ppd

        seg = preds
        pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)
        return 