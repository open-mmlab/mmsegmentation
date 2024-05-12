_base_ = './icnet_r50-d8_4xb2-20k_HOTS_v1-640x480.py'
model = dict(
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnet50_v1c'))))
