# Copyright (c) OpenMMLab. All rights reserved.
import torch

# from mmseg.models import build_loss
from mmseg.models.losses import InverseFormLoss


def test_inverseform_loss():
    pred = torch.zeros(3, 1, 4, 4)
    target = torch.zeros(3, 1, 4, 4)

    for i in range(3):
        pred[i, 0, i, :] = 1
        target[i, 0, :, i] = 1
        print(f'pred is:{pred[i]}, \n target is {target[i]}')

    # test inverseform_loss
    inverseNet_path = './checkpoints/distance_measures_regressor.pth'
    # loss_cfg = dict(type='InverseFormLoss', inverseNet_path=inverseNet_path)
    # inverseform_loss = build_loss(loss_cfg)
    loss_class = InverseFormLoss(inverseNet_path=inverseNet_path)
    loss = loss_class(pred, target)
    assert isinstance(loss, torch.Tensor)
