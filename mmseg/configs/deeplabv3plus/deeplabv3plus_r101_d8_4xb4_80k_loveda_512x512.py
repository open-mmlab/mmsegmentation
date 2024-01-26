from mmengine.config import read_base

with read_base():
    from .deeplabv3plus_r50_d8_4xb4_80k_loveda_512x512 import *

model.update(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
