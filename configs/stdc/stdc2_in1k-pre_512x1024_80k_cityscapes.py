_base_ = './stdc2_512x1024_80k_cityscapes.py'
model = dict(
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(
                type='Pretrained', checkpoint='./pretrained/stdc2.pth'))))
