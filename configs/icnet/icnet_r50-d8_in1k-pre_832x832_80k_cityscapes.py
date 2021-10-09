_base_ = './icnet_r50-d8_832x832_80k_cityscapes.py'
model = dict(
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnet50_v1c'))))
