# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .segformer_mit_b0_8xb2_160k_ade20k_512x512 import *  # noqa: F401,F403

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth'  # noqa

# model settings
model.update(  # noqa: F405
    backbone=dict(
        init_cfg=dict(checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 4, 18, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))
