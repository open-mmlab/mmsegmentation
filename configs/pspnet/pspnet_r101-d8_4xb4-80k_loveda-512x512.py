_base_ = './pspnet_r50-d8_4xb4-80k_loveda-512x512.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnet101_v1c')))
