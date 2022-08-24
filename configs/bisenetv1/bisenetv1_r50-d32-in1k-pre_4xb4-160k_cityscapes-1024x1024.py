_base_ = './bisenetv1_r50-d32_4x4_1024x1024_160k_cityscapes.py'
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnet50_v1c'))))
