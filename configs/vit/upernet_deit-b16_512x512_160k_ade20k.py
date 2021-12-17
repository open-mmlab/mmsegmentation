# Copyright (c) OpenMMLab. All rights reserved.
_base_ = './upernet_vit-b16_mln_512x512_160k_ade20k.py'

model = dict(
    pretrained='pretrain/deit_base_patch16_224-b5f2ef4d.pth',
    backbone=dict(drop_path_rate=0.1),
    neck=None)
