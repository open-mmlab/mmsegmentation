_base_ = './bisenetv1_r101-d32_4xb4-160k_coco-stuff164k-512x512.py'
model = dict(
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnet101_v1c'))))
