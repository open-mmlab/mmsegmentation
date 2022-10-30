_base_ = './maskformer_r50-d32_8xb2-160k_ade20k-512x512.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
