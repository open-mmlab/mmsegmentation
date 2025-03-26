_base_ = './icnet_r50-d8_4xb2-20k_HOTS_v1-640x480.py'
model = dict(
    backbone=dict(
        layer_channels=(128, 512),
        backbone_cfg=dict(
            depth=18,
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnet18_v1c'))))
