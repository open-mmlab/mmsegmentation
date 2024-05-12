_base_ = ['./mask2former_r50_8xb2-20k_HOTS_v1-640x480.py']

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
