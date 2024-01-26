from mmengine.config import read_base

with read_base():
    from .deeplabv3plus_r50_d8_4xb4_80k_loveda_512x512 import *

model.update(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))
