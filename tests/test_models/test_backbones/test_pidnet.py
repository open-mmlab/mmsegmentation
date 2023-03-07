# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile

import torch
from mmengine.registry import init_default_scope

from mmseg.registry import MODELS

init_default_scope('mmseg')


def test_pidnet_backbone():
    # Test PIDNet Standard Forward
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
    model = MODELS.build(backbone_cfg)
    model.init_weights()

    # Test init weights
    temp_file = tempfile.NamedTemporaryFile()
    temp_file.close()
    torch.save(model.state_dict(), temp_file.name)
    backbone_cfg.update(
        init_cfg=dict(type='Pretrained', checkpoint=temp_file.name))
    model = MODELS.build(backbone_cfg)
    model.init_weights()
    os.remove(temp_file.name)

    # Test eval mode
    model.eval()
    batch_size = 1
    imgs = torch.randn(batch_size, 3, 64, 128)
    feats = model(imgs)

    assert type(feats) == torch.Tensor
    assert feats.shape == torch.Size([batch_size, 128, 8, 16])

    # Test train mode
    model.train()
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 64, 128)
    feats = model(imgs)

    assert len(feats) == 3
    # test output for P branch
    assert feats[0].shape == torch.Size([batch_size, 64, 8, 16])
    # test output for I branch
    assert feats[1].shape == torch.Size([batch_size, 128, 8, 16])
    # test output for D branch
    assert feats[2].shape == torch.Size([batch_size, 64, 8, 16])

    # Test pidnet-m
    backbone_cfg.update(channels=64)
    model = MODELS.build(backbone_cfg)
    feats = model(imgs)

    assert len(feats) == 3
    # test output for P branch
    assert feats[0].shape == torch.Size([batch_size, 128, 8, 16])
    # test output for I branch
    assert feats[1].shape == torch.Size([batch_size, 256, 8, 16])
    # test output for D branch
    assert feats[2].shape == torch.Size([batch_size, 128, 8, 16])

    # Test pidnet-l
    backbone_cfg.update(
        channels=64, ppm_channesl=112, num_stem_blocks=3, num_branch_blocks=4)
    model = MODELS.build(backbone_cfg)
    feats = model(imgs)

    assert len(feats) == 3
    # test output for P branch
    assert feats[0].shape == torch.Size([batch_size, 128, 8, 16])
    # test output for I branch
    assert feats[1].shape == torch.Size([batch_size, 256, 8, 16])
    # test output for D branch
    assert feats[2].shape == torch.Size([batch_size, 128, 8, 16])
