# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.backbones.hrformer import (HRFomerModule, HRFormer,
                                             HRFormerBlock)


@pytest.mark.parametrize('block', [HRFormerBlock])
def test_hrformer_module(block):
    # Test multiscale forward
    num_channles = (32, 64)
    num_inchannels = [c * block.expansion for c in num_channles]
    hrmodule = HRFomerModule(
        num_branches=2,
        blocks=block,
        num_inchannels=num_inchannels,
        num_blocks=(4, 4),
        num_channels=num_channles,
    )

    feats = [
        torch.randn(1, num_inchannels[0], 64, 64),
        torch.randn(1, num_inchannels[1], 32, 32)
    ]
    feats = hrmodule(feats)

    assert len(feats) == 2
    assert feats[0].shape == torch.Size([1, num_inchannels[0], 64, 64])
    assert feats[1].shape == torch.Size([1, num_inchannels[1], 32, 32])

    # Test single scale forward
    num_channles = (32, 64)
    in_channels = [c * block.expansion for c in num_channles]
    hrmodule = HRFomerModule(
        num_branches=2,
        blocks=block,
        in_channels=in_channels,
        num_blocks=(4, 4),
        num_channels=num_channles,
        multiscale_output=False,
    )

    feats = [
        torch.randn(1, in_channels[0], 64, 64),
        torch.randn(1, in_channels[1], 32, 32)
    ]
    feats = hrmodule(feats)

    assert len(feats) == 1
    assert feats[0].shape == torch.Size([1, in_channels[0], 64, 64])


def test_hrformer_backbone():
    # only have 3 stages
    extra = dict(
        stage1=dict(
            num_modules=1,
            num_branches=1,
            block='BOTTLENECK',
            num_blocks=(2, ),
            num_channels=(64, )),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block='HRFORMER',
            window_sizes=(7, 7),
            num_heads=(1, 2),
            mlp_ratios=(4, 4),
            num_blocks=(2, 2),
            num_channels=(32, 64)),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block='HRFORMER',
            window_sizes=(7, 7, 7),
            num_heads=(1, 2, 4),
            mlp_ratios=(4, 4, 4),
            num_blocks=(2, 2, 2),
            num_channels=(32, 64, 128)))

    with pytest.raises(AssertionError):
        # HRNet now only support 4 stages
        HRFormer(extra=extra)
    extra['stage4'] = dict(
        num_modules=3,
        num_branches=3,  # should be 4
        block='HRFORMER',
        window_sizes=(7, 7, 7, 7),
        num_heads=(1, 2, 4, 8),
        mlp_ratios=(4, 4, 4, 4),
        num_blocks=(2, 2, 2, 2),
        num_channels=(32, 64, 128, 256))

    with pytest.raises(AssertionError):
        # len(num_blocks) should equal num_branches
        HRFormer(extra=extra)

    extra['stage4']['num_branches'] = 4

    # Test HRFormer-S
    model = HRFormer(extra=extra)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 64, 64)
    feats = model(imgs)
    assert len(feats) == 4
    assert feats[0].shape == torch.Size([1, 32, 16, 16])
    assert feats[3].shape == torch.Size([1, 256, 2, 2])

    # Test single scale output
    model = HRFormer(extra=extra, multiscale_output=False)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 64, 64)
    feats = model(imgs)
    assert len(feats) == 1
    assert feats[0].shape == torch.Size([1, 32, 16, 16])
