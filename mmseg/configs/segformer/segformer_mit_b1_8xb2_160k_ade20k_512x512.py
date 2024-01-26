# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .segformer_mit_b0_8xb2_160k_ade20k_512x512 import *  # noqa: F401,F403

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth'  # noqa

# model settings
model.update(  # noqa: F405
    backbone=dict(
        init_cfg=dict(checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[2, 2, 2, 2]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))
