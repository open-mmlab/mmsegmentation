# Modified from https://github.com/Qualcomm-AI-research/InverseForm
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.models.losses import BoundaryLoss
from mmseg.registry import MODELS

# Most of the code below is from the following repo:
# https://github.com/Qualcomm-AI-research/InverseForm/blob/main/models/InverseForm.py


class InverseNet(nn.Module):
    """Regressor for the 3 * 2 affine matrix"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(224 * 224 * 2, 1000), nn.ReLU(True), nn.Linear(1000, 32),
            nn.ReLU(True), nn.Linear(32, 4))

    def forward(self, x1: Tensor, x2: Tensor) -> (Tensor, Tensor, Tensor):
        """
        Args:
            x1(torch.tensor): pred boundary binary map.
            x2(torch.tensor): true boundary binary map.
        Returns:
            x1(torch.tensor): pred boundary binary map.
            x2(torch.tensor): true boundary binary map.
            self.fc(x)(torch.tensor): 3 * 2 affine matrix.
        """
        x = torch.cat((x1.view(-1, 224 * 224), x2.view(-1, 224 * 224)), dim=1)
        return x1, x2, self.fc(x)


# Most of the code below is from the following repo:
# https://github.com/Qualcomm-AI-research/InverseForm/blob/main/models/loss/utils.py


class InverseFormLoss(nn.Module):
    """InverseTransformNet loss(measuring distances from homography) load
    pretrained InverseForm net and freeze it's weights.

    Args:
        inverseNet_path: the path of pretrained InverseNet,download from
        https://github.com/Qualcomm-AI-research/InverseForm/releases/download/v1.0/distance_measures_regressor.pth
    """

    def __init__(self, inverseNet_path: str):
        super().__init__()
        self.tile_factor = 3
        self.resized_dim = 672
        self.tiled_dim = self.resized_dim // self.tile_factor
        self.INVERSEFORM_MODULE = inverseNet_path

        inversenet_backbone = InverseNet()
        self.inversenet = load_model_from_dict(inversenet_backbone,
                                               self.INVERSEFORM_MODULE)
        for param in self.inversenet.parameters():
            param.requires_grad = False

    def forward(self, inputs: torch.tensor,
                targets: torch.tensor) -> torch.Tensor:
        """
        Args:
            inputs(torch.tensor): pred boundary binary map.
            targets(torch.tensor): true boundary binary map.
        Returns:
            mean_square_inverse_loss(torch.tensor): InverseTransformNet loss.
        """
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


def load_model_from_dict(model: nn.Module, pretrained: str) -> nn.Module:
    """load InverseFormNet pretrained weight
    Args:
        model(nn.Module): model structure.
        pretrained(str): checkpoint path.
    Returns:
        model(nn.Module): inverseNet model with pretrained weight.
    """
    pretrained_dict = torch.load(pretrained)
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
    """Image Weighted Cross Entropy Loss(Segmentation Loss)."""

    def __init__(self,
                 num_classes: int,
                 weight: torch.Tensor = None,
                 ignore_index: int = 255,
                 norm: bool = False,
                 upper_bound: float = 1.0,
                 fp16: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.nll_loss = nn.NLLLoss(weight, ignore_index=ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.fp16 = fp16

    def calculate_weights(self, target: torch.Tensor) -> torch.Tensor:
        """Calculate weights of classes based on the training crop.

        Args:
            target(torch.Tensor): true training crop sample.
        Returns:
            hist(torch.Tensor): class weights.
        """
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

    def forward(self, inputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs(torch.Tensor): pred mask.
            targets(torch.Tensor): true mask.
        Returns:
            loss(torch.Tensor): Segmentation Loss.
        """
        loss = 0.0
        for i in range(0, inputs.shape[0]):
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
class JointInverseFormLoss(nn.Module):
    """loss which are boundary-aware, combined with segmentation loss, balanced
    Cross-Entropy loss and InverseForm loss This loss is proposed in
    `InverseForm: A Loss Function for Structured Boundary-Aware Segmentation`
    https://arxiv.org/abs/2104.02745.

    Args:
    num_classes(int): number of classes
    weight(Tensor): weight (Tensor, optional): a manual rescaling weight given
        to each class. If given, it has to be a Tensor of size `C`.Otherwise,
        it is treated as if having all ones.Used in nn.NLLLoss.
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
    loss_name (str, optional): Name of the loss item. If you want this loss
        item to be included into the backward graph, `loss_` must be the
        prefix of the name. Defaults to 'loss_inverseform'.
    """

    def __init__(self,
                 num_classes: int = 19,
                 weight: torch.Tensor = None,
                 ignore_index: int = 255,
                 norm: bool = False,
                 upper_bound: float = 1.0,
                 fp16: bool = False,
                 edge_weight: float = 0.3,
                 inv_weight: float = 0.3,
                 seg_weight: float = 1,
                 att_weight: float = 0.1,
                 inverseNet_path:
                 str = './checkpoints/distance_measures_regressor.pth',
                 loss_name: str = 'loss_inverseform'):
        super().__init__()
        self.seg_loss = ImageBasedCrossEntropyLoss2d(
            num_classes=num_classes,
            weight=weight,
            ignore_index=ignore_index,
            norm=norm,
            upper_bound=upper_bound,
            fp16=fp16)
        self.inverse_distance = InverseFormLoss(inverseNet_path)
        self.edge_weight = edge_weight
        self.seg_weight = seg_weight
        self.att_weight = att_weight
        self.inv_weight = inv_weight
        self._loss_name = loss_name

    def bce2d(self, input, target):
        """To calculate balanced cross-entropy of two binary boundary map
        input and target are both binary boundary map.
        Args:
            input (Tensor): Predictions of the boundary head.
            target (Tensor): Ground truth of the boundary.

        Returns:
            Tensor: Loss tensor.
        """
        edge_loss = BoundaryLoss(self.edge_weight)
        return edge_loss

    def forward(self, inputs: (torch.Tensor, torch.Tensor),
                targets: (torch.Tensor, torch.Tensor),
                **kwargs) -> torch.Tensor:
        """Forward function.

        Args:
            inputs (Tensor): Predictions of the segmentation head.
                (pred_segmentation_mask, pred_edge_binary_boundary_map)
            target (Tensor): Ground truth of the image.
                (gt_segmentation_mask, gt_edge_binary_boundary_map)
        Returns:
            Tensor: Loss tensor.
        """
        segin, edgein = inputs
        segmask, edgemask = targets

        total_loss = self.seg_weight * self.seg_loss(segin, segmask) \
            + self.edge_weight * self.bce2d(edgein, edgemask) \
            + self.inv_weight * self.inverse_distance(edgein, edgemask)
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
