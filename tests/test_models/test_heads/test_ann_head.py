# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models.decode_heads import ANNHead
from .utils import to_cuda


def test_ann_head():

    inputs = [torch.randn(1, 16, 45, 45), torch.randn(1, 32, 21, 21)]
    head = ANNHead(
        in_channels=[16, 32],
        channels=16,
        num_classes=19,
        in_index=[-2, -1],
        project_channels=8)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 21, 21)
