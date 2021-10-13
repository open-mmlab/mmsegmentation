# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models.decode_heads import ERFHead
from .utils import to_cuda


def test_erf_head():
    head = ERFHead(in_channels=16, channels=19, num_classes=19)
    assert head.output_conv.in_channels == 16
    assert head.output_conv.out_channels == 19

    inputs = torch.randn(1, 16, 45, 45).cuda()
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 90, 90)
