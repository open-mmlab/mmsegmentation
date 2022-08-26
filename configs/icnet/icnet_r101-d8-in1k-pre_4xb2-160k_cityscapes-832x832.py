_base_ = './icnet_r50-d8_4xb2-160k_cityscapes-832x832.py'
model = dict(
    backbone=dict(
        backbone_cfg=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnet101_v1c'))))
