# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.necks import FARNeck


def test_fastfcn_neck():
    # Test FarNeck Standard Forward
    model = FARNeck(
        neck_cfg=dict(
            type='FPN', in_channels=[2, 4, 8, 16], out_channels=2, num_outs=4),
        scene_relation_in_channels=16,
        scene_relation_channel_list=(2, 2, 2, 2),
        scene_relation_out_channels=2,
    )
    model.init_weights()
    model.train()
    batch_size = 1
    input = [
        torch.randn(batch_size, 2, 56, 56),
        torch.randn(batch_size, 4, 28, 28),
        torch.randn(batch_size, 8, 14, 14),
        torch.randn(batch_size, 16, 7, 7),
    ]
    feat = model(input)

    assert len(feat) == 4
    assert feat[0].shape == torch.Size([batch_size, 2, 56, 56])
    assert feat[1].shape == torch.Size([batch_size, 2, 28, 28])
    assert feat[2].shape == torch.Size([batch_size, 2, 14, 14])
    assert feat[3].shape == torch.Size([batch_size, 2, 7, 7])

    with pytest.raises(AssertionError):
        # FARNeck input length constraints.
        input = [
            torch.randn(batch_size, 2, 56, 56),
            torch.randn(batch_size, 4, 28, 28),
            torch.randn(batch_size, 8, 14, 14),
        ]
        model(input)
