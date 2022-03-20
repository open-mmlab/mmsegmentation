# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models.decode_heads import ANNHead
from .utils import to_cuda


def test_ann_head():

    inputs = [torch.randn(1, 4, 45, 45), torch.randn(1, 8, 21, 21)]
    head = ANNHead(
        in_channels=[4, 8],
        channels=2,
        num_classes=19,
        in_index=[-2, -1],
        project_channels=8)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 21, 21)
