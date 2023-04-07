# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.registry import init_default_scope

from mmseg.registry import MODELS


def test_pidnet_head():
    init_default_scope('mmseg')

    # Test PIDNet decode head Standard Forward
    norm_cfg = dict(type='BN', requires_grad=True)
    backbone_cfg = dict(
        type='PIDNet',
        in_channels=3,
        channels=32,
        ppm_channels=96,
        num_stem_blocks=2,
        num_branch_blocks=3,
        align_corners=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True))
    decode_head_cfg = dict(
        type='PIDHead',
        in_channels=128,
        channels=128,
        num_classes=19,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=True,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=[
                    0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                    1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                    1.0865, 1.0955, 1.0865, 1.1529, 1.0507
                ],
                loss_weight=0.4),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=[
                    0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                    1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                    1.0865, 1.0955, 1.0865, 1.1529, 1.0507
                ],
                loss_weight=1.0),
            dict(type='BoundaryLoss', loss_weight=20.0),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=[
                    0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                    1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                    1.0865, 1.0955, 1.0865, 1.1529, 1.0507
                ],
                loss_weight=1.0)
        ])
    backbone = MODELS.build(backbone_cfg)
    head = MODELS.build(decode_head_cfg)

    # Test train mode
    backbone.train()
    head.train()
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 64, 128)
    feats = backbone(imgs)
    seg_logit = head(feats)

    assert isinstance(seg_logit, tuple)
    assert len(seg_logit) == 3

    p_logits, i_logits, d_logits = seg_logit
    assert p_logits.shape == (batch_size, 19, 8, 16)
    assert i_logits.shape == (batch_size, 19, 8, 16)
    assert d_logits.shape == (batch_size, 1, 8, 16)

    # Test eval mode
    backbone.eval()
    head.eval()
    feats = backbone(imgs)
    seg_logit = head(feats)

    assert isinstance(seg_logit, torch.Tensor)
    assert seg_logit.shape == (batch_size, 19, 8, 16)
