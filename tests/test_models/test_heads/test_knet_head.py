# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models.decode_heads.knet_head import (IterativeDecodeHead,
                                                 KernelUpdateHead)
from .utils import to_cuda

num_stages = 3
conv_kernel_size = 1

kernel_updator_cfg = dict(
    type='KernelUpdator',
    in_channels=16,
    feat_channels=16,
    out_channels=16,
    gate_norm_act=True,
    activate_out=True,
    act_cfg=dict(type='ReLU', inplace=True),
    norm_cfg=dict(type='LN'))


def test_knet_head():
    # test init function of kernel update head
    kernel_update_head = KernelUpdateHead(
        num_classes=150,
        num_ffn_fcs=2,
        num_heads=8,
        num_mask_fcs=1,
        feedforward_channels=128,
        in_channels=32,
        out_channels=32,
        dropout=0.0,
        conv_kernel_size=conv_kernel_size,
        ffn_act_cfg=dict(type='ReLU', inplace=True),
        with_ffn=True,
        feat_transform_cfg=dict(conv_cfg=dict(type='Conv2d'), act_cfg=None),
        kernel_init=True,
        kernel_updator_cfg=kernel_updator_cfg)
    kernel_update_head.init_weights()

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
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(
                    conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_init=False,
                kernel_updator_cfg=kernel_updator_cfg)
            for _ in range(num_stages)
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

    # test whether only return the prediction of
    # the last stage during testing
    with torch.no_grad():
        head.eval()
        outputs = head(inputs)
        assert outputs.shape == (1, head.num_classes, 27, 16)

    # test K-Net without `feat_transform_cfg`
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
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=None,
                kernel_updator_cfg=kernel_updator_cfg)
            for _ in range(num_stages)
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

    # test K-Net with
    # self.mask_transform_stride == 2 and self.feat_gather_stride == 1
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
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(
                    conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_init=False,
                mask_transform_stride=2,
                feat_gather_stride=1,
                kernel_updator_cfg=kernel_updator_cfg)
            for _ in range(num_stages)
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
    assert outputs[-1].shape == (1, head.num_classes, 26, 16)

    # test loss function in K-Net
    fake_label = torch.ones_like(
        outputs[-1][:, 0:1, :, :], dtype=torch.int16).long()
    loss = head.losses(seg_logit=outputs, seg_label=fake_label)
    assert loss['loss_ce.s0'] != torch.zeros_like(loss['loss_ce.s0'])
    assert loss['loss_ce.s1'] != torch.zeros_like(loss['loss_ce.s1'])
    assert loss['loss_ce.s2'] != torch.zeros_like(loss['loss_ce.s2'])
    assert loss['loss_ce.s3'] != torch.zeros_like(loss['loss_ce.s3'])
