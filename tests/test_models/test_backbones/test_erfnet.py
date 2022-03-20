# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.backbones import ERFNet
from mmseg.models.backbones.erfnet import (DownsamplerBlock, NonBottleneck1d,
                                           UpsamplerBlock)


def test_erfnet_backbone():
    # Test ERFNet Standard Forward.
    model = ERFNet(
        in_channels=3,
        enc_downsample_channels=(16, 64, 128),
        enc_stage_non_bottlenecks=(5, 8),
        enc_non_bottleneck_dilations=(2, 4, 8, 16),
        enc_non_bottleneck_channels=(64, 128),
        dec_upsample_channels=(64, 16),
        dec_stages_non_bottleneck=(2, 2),
        dec_non_bottleneck_channels=(64, 16),
        dropout_ratio=0.1,
    )
    model.init_weights()
    model.train()
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 256, 512)
    output = model(imgs)

    # output for segment Head
    assert output[0].shape == torch.Size([batch_size, 16, 128, 256])

    # Test input with rare shape
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 527, 279)
    output = model(imgs)
    assert len(output[0]) == batch_size

    with pytest.raises(AssertionError):
        # Number of encoder downsample block and decoder upsample block.
        ERFNet(
            in_channels=3,
            enc_downsample_channels=(16, 64, 128),
            enc_stage_non_bottlenecks=(5, 8),
            enc_non_bottleneck_dilations=(2, 4, 8, 16),
            enc_non_bottleneck_channels=(64, 128),
            dec_upsample_channels=(128, 64, 16),
            dec_stages_non_bottleneck=(2, 2),
            dec_non_bottleneck_channels=(64, 16),
            dropout_ratio=0.1,
        )
    with pytest.raises(AssertionError):
        # Number of encoder downsample block and encoder Non-bottleneck block.
        ERFNet(
            in_channels=3,
            enc_downsample_channels=(16, 64, 128),
            enc_stage_non_bottlenecks=(5, 8, 10),
            enc_non_bottleneck_dilations=(2, 4, 8, 16),
            enc_non_bottleneck_channels=(64, 128),
            dec_upsample_channels=(64, 16),
            dec_stages_non_bottleneck=(2, 2),
            dec_non_bottleneck_channels=(64, 16),
            dropout_ratio=0.1,
        )
    with pytest.raises(AssertionError):
        # Number of encoder downsample block and
        # channels of encoder Non-bottleneck block.
        ERFNet(
            in_channels=3,
            enc_downsample_channels=(16, 64, 128),
            enc_stage_non_bottlenecks=(5, 8),
            enc_non_bottleneck_dilations=(2, 4, 8, 16),
            enc_non_bottleneck_channels=(64, 128, 256),
            dec_upsample_channels=(64, 16),
            dec_stages_non_bottleneck=(2, 2),
            dec_non_bottleneck_channels=(64, 16),
            dropout_ratio=0.1,
        )

    with pytest.raises(AssertionError):
        # Number of encoder Non-bottleneck block and number of its channels.
        ERFNet(
            in_channels=3,
            enc_downsample_channels=(16, 64, 128),
            enc_stage_non_bottlenecks=(5, 8, 3),
            enc_non_bottleneck_dilations=(2, 4, 8, 16),
            enc_non_bottleneck_channels=(64, 128),
            dec_upsample_channels=(64, 16),
            dec_stages_non_bottleneck=(2, 2),
            dec_non_bottleneck_channels=(64, 16),
            dropout_ratio=0.1,
        )
    with pytest.raises(AssertionError):
        # Number of decoder upsample block and decoder Non-bottleneck block.
        ERFNet(
            in_channels=3,
            enc_downsample_channels=(16, 64, 128),
            enc_stage_non_bottlenecks=(5, 8),
            enc_non_bottleneck_dilations=(2, 4, 8, 16),
            enc_non_bottleneck_channels=(64, 128),
            dec_upsample_channels=(64, 16),
            dec_stages_non_bottleneck=(2, 2, 3),
            dec_non_bottleneck_channels=(64, 16),
            dropout_ratio=0.1,
        )
    with pytest.raises(AssertionError):
        # Number of decoder Non-bottleneck block and number of its channels.
        ERFNet(
            in_channels=3,
            enc_downsample_channels=(16, 64, 128),
            enc_stage_non_bottlenecks=(5, 8),
            enc_non_bottleneck_dilations=(2, 4, 8, 16),
            enc_non_bottleneck_channels=(64, 128),
            dec_upsample_channels=(64, 16),
            dec_stages_non_bottleneck=(2, 2),
            dec_non_bottleneck_channels=(64, 16, 8),
            dropout_ratio=0.1,
        )


def test_erfnet_downsampler_block():
    x_db = DownsamplerBlock(16, 64)
    assert x_db.conv.in_channels == 16
    assert x_db.conv.out_channels == 48
    assert len(x_db.bn.weight) == 64
    assert x_db.pool.kernel_size == 2
    assert x_db.pool.stride == 2


def test_erfnet_non_bottleneck_1d():
    x_nb1d = NonBottleneck1d(16, 0, 1)
    assert x_nb1d.convs_layers[0].in_channels == 16
    assert x_nb1d.convs_layers[0].out_channels == 16
    assert x_nb1d.convs_layers[2].in_channels == 16
    assert x_nb1d.convs_layers[2].out_channels == 16
    assert x_nb1d.convs_layers[5].in_channels == 16
    assert x_nb1d.convs_layers[5].out_channels == 16
    assert x_nb1d.convs_layers[7].in_channels == 16
    assert x_nb1d.convs_layers[7].out_channels == 16
    assert x_nb1d.convs_layers[9].p == 0


def test_erfnet_upsampler_block():
    x_ub = UpsamplerBlock(64, 16)
    assert x_ub.conv.in_channels == 64
    assert x_ub.conv.out_channels == 16
    assert len(x_ub.bn.weight) == 16
