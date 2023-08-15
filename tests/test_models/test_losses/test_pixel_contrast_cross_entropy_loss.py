# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.losses import PixelContrastCrossEntropyLoss
from mmseg.models.decode_heads import HRNetContrastHead
import torch.nn as nn

def test_pixel_contrast_crossentropy_loss():
    # temperature should >=0 and base_temperature should >0
    with pytest.raises(KeyError):
        PixelContrastCrossEntropyLoss(base_temperature=0)
    with pytest.raises(KeyError):
        PixelContrastCrossEntropyLoss(temperature=-0.1)

    # ignore_index should be an int between 0 and 255
    with pytest.raises(KeyError):
        PixelContrastCrossEntropyLoss(ignore_index=256)
    with pytest.raises(KeyError):
        PixelContrastCrossEntropyLoss(ignore_index=[255])
    
    # max_samples should be an int and >=0
    with pytest.raises(KeyError):
        PixelContrastCrossEntropyLoss(max_samples=-1)
    with pytest.raises(KeyError):
        PixelContrastCrossEntropyLoss(max_samples=0.1)

    # max_views should be an int and >=0
    with pytest.raises(KeyError):
        PixelContrastCrossEntropyLoss(max_views=-1)
    with pytest.raises(KeyError):
        PixelContrastCrossEntropyLoss(max_views=0.1)

    target = torch.rand((1, 23, 23))
    inputs = [torch.randn(1, 8, 23, 23)]
    head = HRNetContrastHead(in_channels=8, channels=4, num_classes=19, proj_mode='linear')
    loss = PixelContrastCrossEntropyLoss()
    with pytest.raises(AssertionError):
        # pred must be a dict that output from HRNetContrastHead
        pred = torch.randn(1, 8, 23, 23)
        loss(pred, target)

    pred = head(inputs)
    loss = loss(pred, target)
    assert isinstance(loss, torch.Tensor)



    head = HRNetContrastHead(in_channels=8, channels=4, num_classes=19, proj_mode='convmlp')
    loss = PixelContrastCrossEntropyLoss()
    with pytest.raises(AssertionError):
        # pred must be a dict that output from HRNetContrastHead
        pred = torch.randn(1, 8, 23, 23)
        loss(pred, target)

    pred = head(inputs)
    loss = loss(pred, target)
    assert isinstance(loss, torch.Tensor)



    target = torch.rand((1, 23, 23))
    inputs = [torch.randn(1, 10, 23, 23)]
    head = HRNetContrastHead(in_channels=10, channels=6, num_classes=19, proj_mode='convmlp')
    loss = PixelContrastCrossEntropyLoss()

    pred = head(inputs)
    loss = loss(pred, target)
    assert isinstance(loss, torch.Tensor)

