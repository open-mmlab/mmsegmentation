# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.decode_heads import FARHead
from .utils import to_cuda


def test_far_head():

    # test R50-D32 feature map
    inputs = [
        torch.randn(1, 4, 16, 16),
        torch.randn(1, 4, 8, 8),
        torch.randn(1, 4, 4, 4),
        torch.randn(1, 4, 2, 2),
    ]
    head = FARHead(
        in_channels=4,
        out_channels=2,
        in_feat_output_strides=(4, 8, 16, 32),
        out_feat_output_stride=4,
        num_classes=16)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 16, 16)

    # test R50-D8 feature map
    inputs = [
        torch.randn(1, 4, 16, 16),
        torch.randn(1, 4, 8, 8),
        torch.randn(1, 4, 8, 8),
        torch.randn(1, 4, 8, 8),
    ]
    head = FARHead(
        in_channels=4,
        out_channels=2,
        in_feat_output_strides=(4, 8, 8, 8),
        out_feat_output_stride=4,
        num_classes=16)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 16, 16)

    with pytest.raises(AssertionError):
        # FARHead input and output stride constraints.

        # decoder head for R50-D32 feature map
        head = FARHead(
            in_channels=4,
            out_channels=2,
            in_feat_output_strides=(4, 8, 16, 32),
            out_feat_output_stride=4,
            num_classes=16)

        # R50-D8 feature map
        inputs = [
            torch.randn(1, 4, 16, 16),
            torch.randn(1, 4, 8, 8),
            torch.randn(1, 4, 8, 8),
            torch.randn(1, 4, 8, 8),
        ]
        head(inputs)

    # test annealing softmax focalloss
    fake_label = torch.ones_like(
        outputs[:, 0:1, :, :], dtype=torch.int16).long()

    head.annealing_type = 'cosine'
    loss = head.losses(seg_logit=outputs, seg_label=fake_label)
    assert loss['loss_asfocal'] != torch.zeros_like(loss['loss_asfocal'])

    head.annealing_type = 'poly'
    loss = head.losses(seg_logit=outputs, seg_label=fake_label)
    assert loss['loss_asfocal'] != torch.zeros_like(loss['loss_asfocal'])

    head.annealing_type = 'linear'
    loss = head.losses(seg_logit=outputs, seg_label=fake_label)
    assert loss['loss_asfocal'] != torch.zeros_like(loss['loss_asfocal'])
