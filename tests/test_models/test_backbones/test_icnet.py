# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.backbones import ICNet


def test_icnet_backbone():
    with pytest.raises(TypeError):
        # Must give backbone dict in config file.
        ICNet(
            in_channels=3,
            layer_channels=(128, 512),
            light_branch_middle_channels=8,
            psp_out_channels=128,
            out_channels=(16, 128, 128),
            backbone_cfg=None)

    # Test ICNet Standard Forward
    model = ICNet(
        layer_channels=(128, 512),
        backbone_cfg=dict(
            type='ResNetV1c',
            in_channels=3,
            depth=18,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            style='pytorch',
            contract_dilation=True),
    )
    assert hasattr(model.backbone,
                   'maxpool') and model.backbone.maxpool.ceil_mode is True
    model.init_weights()
    model.train()
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 32, 64)
    feat = model(imgs)

    assert model.psp_modules[0][0].output_size == 1
    assert model.psp_modules[1][0].output_size == 2
    assert model.psp_modules[2][0].output_size == 3
    assert model.psp_bottleneck.padding == 1
    assert model.conv_sub1[0].padding == 1

    assert len(feat) == 3
    assert feat[0].shape == torch.Size([batch_size, 64, 4, 8])
