# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models.decode_heads import LightHamHead
from .utils import _conv_has_norm, to_cuda

ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)


def test_ham_head():

    # test without sync_bn
    head = LightHamHead(
        in_channels=[16, 32, 64],
        in_index=[1, 2, 3],
        channels=64,
        ham_channels=64,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        ham_kwargs=dict(
            MD_S=1,
            MD_R=64,
            train_steps=6,
            eval_steps=7,
            inv_t=100,
            rand_init=True))
    assert not _conv_has_norm(head, sync_bn=False)

    inputs = [
        torch.randn(1, 8, 32, 32),
        torch.randn(1, 16, 16, 16),
        torch.randn(1, 32, 8, 8),
        torch.randn(1, 64, 4, 4)
    ]
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert head.in_channels == [16, 32, 64]
    assert head.hamburger.ham_in.in_channels == 64
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 16, 16)
