# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn

from mmseg.models.decode_heads import HRNetContrastHead
from .utils import to_cuda


def test_hrnet_contrast_head():
    with pytest.raises(KeyError):
        # proj_n must >0
        HRNetContrastHead(in_channels=8, channels=4, num_classes=19, proj_n=0)

    with pytest.raises(KeyError):
        # proj_mode must in ['convmlp','linear']
        HRNetContrastHead(
            in_channels=8, channels=4, num_classes=19, proj_mode='ploy')

    with pytest.raises(KeyError):
        # drop_p must >=0 and should be a float
        HRNetContrastHead(
            in_channels=8, channels=4, num_classes=19, drop_p=-0.1)

    HRNetContrastHead(in_channels=8, channels=4, num_classes=19, drop_p=0.0)
    HRNetContrastHead(in_channels=8, channels=4, num_classes=19, drop_p=1.0)

    # test proj_n=256
    inputs = [torch.randn(1, 8, 23, 23)]
    head = HRNetContrastHead(
        in_channels=8, channels=4, num_classes=19, proj_mode='convmlp')
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert isinstance(head.projhead.proj, nn.Sequential)
    outputs = head(inputs)
    assert outputs['proj'].shape == (1, head.proj_n, 23, 23)
    assert outputs['seg'].shape == (1, head.num_classes, 23, 23)

    # test proj_mode='convmlp'
    inputs = [torch.randn(1, 8, 23, 23)]
    head = HRNetContrastHead(
        in_channels=8, channels=4, num_classes=19, proj_mode='linear')
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert isinstance(head.projhead.proj, nn.Conv2d)
    outputs = head(inputs)
    assert outputs['proj'].shape == (1, head.proj_n, 23, 23)
    assert outputs['seg'].shape == (1, head.num_classes, 23, 23)
