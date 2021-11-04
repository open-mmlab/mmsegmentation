# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import ConvModule

from mmseg.models.backbones import BiSeNetV2
from mmseg.models.backbones.bisenetv2 import (BGALayer, DetailBranch,
                                              SemanticBranch)


def test_bisenetv2_backbone():
    # Test BiSeNetV2 Standard Forward
    model = BiSeNetV2()
    model.init_weights()
    model.train()
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 128, 256)
    feat = model(imgs)

    assert len(feat) == 5
    # output for segment Head
    assert feat[0].shape == torch.Size([batch_size, 128, 16, 32])
    # for auxiliary head 1
    assert feat[1].shape == torch.Size([batch_size, 16, 32, 64])
    # for auxiliary head 2
    assert feat[2].shape == torch.Size([batch_size, 32, 16, 32])
    # for auxiliary head 3
    assert feat[3].shape == torch.Size([batch_size, 64, 8, 16])
    # for auxiliary head 4
    assert feat[4].shape == torch.Size([batch_size, 128, 4, 8])

    # Test input with rare shape
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 95, 27)
    feat = model(imgs)
    assert len(feat) == 5


def test_bisenetv2_DetailBranch():
    x = torch.randn(1, 3, 32, 64)
    detail_branch = DetailBranch(detail_channels=(64, 16, 32))
    assert isinstance(detail_branch.detail_branch[0][0], ConvModule)
    x_out = detail_branch(x)
    assert x_out.shape == torch.Size([1, 32, 4, 8])


def test_bisenetv2_SemanticBranch():
    semantic_branch = SemanticBranch(semantic_channels=(16, 32, 64, 128))
    assert semantic_branch.stage1.pool.stride == 2


def test_bisenetv2_BGALayer():
    x_a = torch.randn(1, 8, 8, 16)
    x_b = torch.randn(1, 8, 2, 4)
    bga = BGALayer(out_channels=8)
    assert isinstance(bga.conv, ConvModule)
    x_out = bga(x_a, x_b)
    assert x_out.shape == torch.Size([1, 8, 8, 16])
