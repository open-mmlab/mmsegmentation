# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.backbones import ICNet


def test_icnet_backbone():
    with pytest.raises(TypeError):
        # Must give backbone dict in config file.
        ICNet(
            in_channels=3,
            layer_channels=(512, 2048),
            light_branch_middle_channels=32,
            psp_out_channels=512,
            out_channels=(64, 256, 256),
            backbone_cfg=None)

    # Test ICNet Standard Forward
    model = ICNet(
        backbone_cfg=dict(
            type='ResNetV1c',
            in_channels=3,
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            style='pytorch',
            contract_dilation=True), )
    model.init_weights()
    model.train()
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 512, 1024)
    feat = model(imgs)

    assert len(feat) == 3
    assert feat[0].shape == torch.Size([batch_size, 64, 64, 128])
