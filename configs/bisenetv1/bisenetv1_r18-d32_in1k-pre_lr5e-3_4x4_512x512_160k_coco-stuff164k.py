_base_ = './bisenetv1_r18-d32_lr5e-3_4x4_512x512_160k_coco-stuff164k.py'
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnet18_v1c'))),
)
