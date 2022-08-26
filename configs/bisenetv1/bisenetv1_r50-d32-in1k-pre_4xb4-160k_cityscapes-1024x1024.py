_base_ = './bisenetv1_r50-d32_4xb4-160k_cityscapes-1024x1024.py'
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnet50_v1c'))))
