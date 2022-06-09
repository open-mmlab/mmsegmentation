_base_ = './pspnet_r50-d8_512x512_80k_loveda.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnet101_v1c')))
