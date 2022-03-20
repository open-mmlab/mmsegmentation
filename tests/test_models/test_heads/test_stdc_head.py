# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models.decode_heads import STDCHead
from .utils import to_cuda


def test_stdc_head():
    inputs = [torch.randn(1, 32, 21, 21)]
    head = STDCHead(
        in_channels=32,
        channels=8,
        num_convs=1,
        num_classes=2,
        in_index=-1,
        loss_decode=[
            dict(
                type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
        ])
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert isinstance(outputs, torch.Tensor) and len(outputs) == 1
    assert outputs.shape == torch.Size([1, head.num_classes, 21, 21])

    fake_label = torch.ones_like(
        outputs[:, 0:1, :, :], dtype=torch.int16).long()
    loss = head.losses(seg_logit=outputs, seg_label=fake_label)
    assert loss['loss_ce'] != torch.zeros_like(loss['loss_ce'])
    assert loss['loss_dice'] != torch.zeros_like(loss['loss_dice'])
