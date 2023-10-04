# Modified from https://github.com/Qualcomm-AI-research/InverseForm
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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


def load_model_from_dict(model: nn.Module,
                         pretrained: str,
                         map_location: str = None) -> nn.Module:
    """load InverseFormNet pretrained weight
    Args:
        model(nn.Module): model structure.
        pretrained(str): checkpoint path.
    Returns:
        model(nn.Module): inverseNet model with pretrained weight.
    """
    pretrained_dict = torch.load(pretrained, map_location=map_location)
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


@MODELS.register_module()
class InverseFormLoss(nn.Module):
    """InverseForm loss, https://arxiv.org/abs/2104.02745.

    Args:
        tile_factor(int): divide the image to several tile, Default 3.
        resized_dim(int): resize input and output to
            (resized_dim,2*resized_dim), Default 672.
        inverseNet_path(str): the path of pretrained InverseNet, download from
            https://github.com/Qualcomm-AI-research/InverseForm/releases/download/v1.0/distance_measures_regressor.pth
        map_location(str): If you want to load pretrained model to cpu,
            using `cpu`, Default None, using in gpu.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_inverseform'.
    """

    def __init__(self,
                 tile_factor: int = 3,
                 resized_dim: int = 672,
                 inverseNet_path:
                 str = './checkpoints/distance_measures_regressor.pth',
                 map_location: str = None,
                 loss_name: str = 'loss_inverseform'):
        super().__init__()
        self.tile_factor = tile_factor
        self.resized_dim = resized_dim
        self.tiled_dim = self.resized_dim // self.tile_factor
        self.INVERSEFORM_MODULE = inverseNet_path

        inversenet_backbone = InverseNet()
        self.inversenet = load_model_from_dict(inversenet_backbone,
                                               self.INVERSEFORM_MODULE,
                                               map_location)
        for param in self.inversenet.parameters():
            param.requires_grad = False

    def forward(self, inputs: torch.tensor, targets: torch.tensor,
                **kwargs) -> torch.Tensor:
        """
        Args:
            inputs(torch.tensor): pred boundary binary map.
            targets(torch.tensor): true boundary binary map.
        Returns:
            mean_square_inverse_loss(torch.tensor): InverseTransformNet loss.
        """
        assert inputs.shape == targets.shape
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
