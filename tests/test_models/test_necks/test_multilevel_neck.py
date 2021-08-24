# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models import MultiLevelNeck


def test_multilevel_neck():

    # Test init_weights
    MultiLevelNeck([266], 256).init_weights()

    # Test multi feature maps
    in_channels = [256, 512, 1024, 2048]
    inputs = [torch.randn(1, c, 14, 14) for i, c in enumerate(in_channels)]

    neck = MultiLevelNeck(in_channels, 256)
    outputs = neck(inputs)
    assert outputs[0].shape == torch.Size([1, 256, 7, 7])
    assert outputs[1].shape == torch.Size([1, 256, 14, 14])
    assert outputs[2].shape == torch.Size([1, 256, 28, 28])
    assert outputs[3].shape == torch.Size([1, 256, 56, 56])

    # Test one feature map
    in_channels = [768]
    inputs = [torch.randn(1, 768, 14, 14)]

    neck = MultiLevelNeck(in_channels, 256)
    outputs = neck(inputs)
    assert outputs[0].shape == torch.Size([1, 256, 7, 7])
    assert outputs[1].shape == torch.Size([1, 256, 14, 14])
    assert outputs[2].shape == torch.Size([1, 256, 28, 28])
    assert outputs[3].shape == torch.Size([1, 256, 56, 56])
