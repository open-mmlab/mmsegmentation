# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models.decode_heads.knet import IterativeDecodeHead
from .utils import to_cuda

num_stages = 3
conv_kernel_size = 1


def test_knet_head():
    head = IterativeDecodeHead(
        num_stages=num_stages,
        kernel_update_head=[
            dict(
                type='KernelUpdateHead',
                num_classes=150,
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                feedforward_channels=128,
                in_channels=32,
                out_channels=32,
                dropout=0.0,
                conv_kernel_size=conv_kernel_size,
                mask_upsample_stride=2,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(
                    conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=16,
                    feat_channels=16,
                    out_channels=16,
                    input_feat_shape=3,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN'))) for _ in range(num_stages)
        ],
        kernel_generate_head=dict(
            type='FCNHead',
            in_channels=128,
            in_index=3,
            channels=32,
            num_convs=2,
            concat_input=True,
            dropout_ratio=0.1,
            num_classes=150,
            align_corners=False))
    head.init_weights()

    inputs = [
        torch.randn(1, 16, 27, 32),
        torch.randn(1, 32, 27, 16),
        torch.randn(1, 64, 27, 16),
        torch.randn(1, 128, 27, 16)
    ]
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs[-1].shape == (1, head.num_classes, 27, 16)
