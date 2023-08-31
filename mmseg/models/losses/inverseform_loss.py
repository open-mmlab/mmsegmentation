# Modified from https://github.com/Qualcomm-AI-research/InverseForm
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


class InverseNet(nn.Module):
    """Most of the code below is from the following repo:

    https://github.com/Qualcomm-AI-research/InverseForm/blob/main/models/InverseForm.py
    """

    def __init__(self):
        super().__init__()
        # Regressor for the 3 * 2 affine matrix
        self.fc = nn.Sequential(
            nn.Linear(224 * 224 * 2, 1000), nn.ReLU(True), nn.Linear(1000, 32),
            nn.ReLU(True), nn.Linear(32, 4))

    def forward(self, x1, x2):
        # Perform the usual forward pass
        x = torch.cat((x1.view(-1, 224 * 224), x2.view(-1, 224 * 224)), dim=1)
        return x1, x2, self.fc(x)


# Most of the code below is from the following repo:
# https://github.com/Qualcomm-AI-research/InverseForm/blob/main/models/loss/utils.py

INVERSEFORM_MODULE = os.path.join('checkpoints',
                                  'distance_measures_regressor.pth')


class InverseTransform2D(nn.Module):
    """InverseTransformNet loss(measuring distances from homography) load
    pretrained InverseForm net and freeze it's weights."""

    def __init__(self, model_output=None):
        super().__init__()
        # Setting up loss
        self.tile_factor = 3
        self.resized_dim = 672
        self.tiled_dim = self.resized_dim // self.tile_factor

        inversenet_backbone = InverseNet()
        self.inversenet = load_model_from_dict(inversenet_backbone,
                                               INVERSEFORM_MODULE)
        for param in self.inversenet.parameters():
            param.requires_grad = False

    def forward(self, inputs, targets):
        inputs = F.log_softmax(inputs)

        inputs = F.interpolate(
            inputs,
            size=(self.resized_dim, 2 * self.resized_dim),
            mode='bilinear')
        targets = F.interpolate(
            targets,
            size=(self.resized_dim, 2 * self.resized_dim),
            mode='bilinear')

        tiled_inputs = inputs[:, :, :self.tiled_dim, :self.tiled_dim]
        tiled_targets = targets[:, :, :self.tiled_dim, :self.tiled_dim]
        k = 1
        for i in range(0, self.tile_factor):
            for j in range(0, 2 * self.tile_factor):
                if i + j != 0:
                    tiled_targets = torch.cat(
                        (tiled_targets,
                         targets[:, :, self.tiled_dim * i:self.tiled_dim *
                                 (i + 1), self.tiled_dim * j:self.tiled_dim *
                                 (j + 1)]),
                        dim=0)
                    k += 1

        k = 1
        for i in range(0, self.tile_factor):
            for j in range(0, 2 * self.tile_factor):
                if i + j != 0:
                    tiled_inputs = torch.cat(
                        (tiled_inputs,
                         inputs[:, :, self.tiled_dim * i:self.tiled_dim *
                                (i + 1), self.tiled_dim * j:self.tiled_dim *
                                (j + 1)]),
                        dim=0)
                k += 1

        _, _, distance_coeffs = self.inversenet(tiled_inputs, tiled_targets)

        mean_square_inverse_loss = (((distance_coeffs * distance_coeffs).sum(
            dim=1))**0.5).mean()
        return mean_square_inverse_loss


def load_model_from_dict(model, pretrained):
    """
    load InverseFormNet
    Args:
    model: model framework
    pretrained: checkpoint path
    """
    pretrained_dict = torch.load(pretrained)
    # https://pytorch.org/docs/stable/generated/torch.load.html
    model_dict = model.state_dict()
    updated_model_dict = {}
    for k_model, v_model in model_dict.items():
        if k_model.startswith('model') or k_model.startswith('module'):
            k_updated = '.'.join(k_model.split('.')[1:])
            updated_model_dict[k_updated] = k_model
        else:
            updated_model_dict[k_model] = k_model

    updated_pretrained_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('model') or k.startswith('modules'):
            k = '.'.join(k.split('.')[1:])
        if k in updated_model_dict.keys() and model_dict[k].shape == v.shape:
            updated_pretrained_dict[updated_model_dict[k]] = v

    model_dict.update(updated_pretrained_dict)
    model.load_state_dict(model_dict)
    return model


class ImageBasedCrossEntropyLoss2d(nn.Module):
    """Image Weighted Cross Entropy Loss(no inverseform loss) Do not need to
    convey weight by param, but will calculate by target batch."""

    def __init__(self,
                 classes,
                 weight=None,
                 ignore_index=255,
                 norm=False,
                 reduction='mean',
                 upper_bound=1.0,
                 fp16=False):
        super().__init__()
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss(
            weight, reduction, ignore_index=ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = False  # self.batch_weights = cfg.BATCH_WEIGHTING
        # https://github.com/Qualcomm-AI-research/InverseForm/blob/main/utils/config.py#L54C1-L54C1
        self.fp16 = fp16

    def calculate_weights(self, target):
        """Calculate weights of classes based on the training crop."""
        bins = torch.histc(
            target, bins=self.num_classes, min=0.0, max=self.num_classes)
        hist_norm = bins.float() / bins.sum()
        if self.norm:
            hist = ((bins != 0).float() * self.upper_bound *
                    (1 / hist_norm)) + 1.0
        else:
            hist = ((bins != 0).float() * self.upper_bound *
                    (1. - hist_norm)) + 1.0
        return hist

    def forward(self, inputs, targets, do_rmi=None):
        if self.batch_weights:
            weights = self.calculate_weights(targets)
            self.nll_loss.weight = weights

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculate_weights(targets)
                if self.fp16:
                    weights = weights.half()
                self.nll_loss.weight = weights

            loss += self.nll_loss(
                F.log_softmax(inputs[i].unsqueeze(0), dim=1),
                targets[i].unsqueeze(0),
            )
        return loss


@MODELS.register_module()
class InverseFormLoss(nn.Module):
    """loss which are boundary-aware, combined with segmentation loss, balanced
    Cross-Entropy loss and InverseForm loss This loss is proposed in
    `InverseForm: A Loss Function for Structured Boundary-Aware Segmentation`
    https://arxiv.org/abs/2104.02745.

    Args:
    classes(int): number of classes
    weight(Tensor): weight (Tensor, optional): a manual rescaling weight given
            to each class. If given, it has to be a Tensor of size `C`.
            Otherwise, it is treated as if having all ones.
            Used in nn.NLLLoss.
    reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``.
            Used in nn.NLLLoss
    ignore_index(int): Specifies a target value that is ignored
            and does not contribute to the input gradient, Default: 255.
    norm(bool): whether use norm in calculating batch classes weights,
            Default: False.
    upper_bound(float): The upper bound of weights if calculating weights
            for every classes. Default: 1.0.
    fp16(bool): whether to use precision=16 when training, Default: False.
    edge_weight(float):  weight for balanced cross-entropy loss which
            is weighted cross-entropy loss,Default: 0.3.
    inv_weight(float): weight for inverseform loss, Default: 0.3.
    seg_weight(float): weight for segmentation loss which is corss-entropy
            loss, Default: 1.
    att_weight(float): weight for edge_attention, which were not mentioned
            in paper, Default: 0.1.
    loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_inverseform'.
    """

    def __init__(self,
                 classes,
                 weight=None,
                 reduction='mean',
                 ignore_index=255,
                 norm=False,
                 upper_bound=1.0,
                 fp16=False,
                 edge_weight=0.3,
                 inv_weight=0.3,
                 seg_weight=1,
                 att_weight=0.1,
                 loss_name='loss_inverseform'):
        super().__init__()
        self.num_classes = classes
        self.seg_loss = ImageBasedCrossEntropyLoss2d(
            classes=classes,
            weight=weight,
            reduction=reduction,
            ignore_index=ignore_index,
            norm=norm,
            upper_bound=upper_bound,
            fp16=fp16).cuda()
        self.inverse_distance = InverseTransform2D()
        self.edge_weight = edge_weight
        self.seg_weight = seg_weight
        self.att_weight = att_weight
        self.inv_weight = inv_weight
        self._loss_name = loss_name

    def bce2d(self, input, target):
        """
        To calculate weighted cross-entropy of two binary boundary map
        input and target are both binary boundary map
        Args:
        input: pred binary boundary map
        target: ground truth binary boundary map generated by using Sobel
                filter and threshold on ground truth segmentation mask
        """
        n, c, h, w = input.size()

        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1,
                                    2).transpose(2,
                                                 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)
        ignore_index = (target_t > 1)

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index = ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        weight[ignore_index] = 0

        weight = torch.from_numpy(weight)
        weight = weight
        loss = F.binary_cross_entropy_with_logits(
            log_p, target_t, weight, size_average=True)
        return loss

    def edge_attention(self, input, target, edge):
        n, c, h, w = input.size()
        filler = torch.ones_like(target) * 255
        return self.seg_loss(input,
                             torch.where(edge.max(1)[0] > 0.8, target, filler))

    def forward(self, inputs, targets, do_rmi=None):
        """
        inputs: (pred_segmentation_mask, pred_edge_binary_boundary_map)
        targets: (gt_segmentation_mask, gt_edge_binary_boundary_map)
        """
        segin, edgein = inputs
        segmask, edgemask = targets

        total_loss = self.seg_weight * self.seg_loss(segin, segmask, do_rmi) \
            + self.edge_weight * self.bce2d(edgein, edgemask) \
            + self.att_weight * self.edge_attention(
                segin, segmask, edgein) + \
            self.inv_weight * self.inverse_distance(edgein, edgemask)
        return total_loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
