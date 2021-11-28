_base_ = './stdc-813_512x1024_80k_cityscapes.py'
model = dict(
    backbone=dict(
        stdc_cfg=dict(
            init_cfg=dict(
                type='Pretrained', checkpoint='./pretrained/stdc-813.pth'))))
