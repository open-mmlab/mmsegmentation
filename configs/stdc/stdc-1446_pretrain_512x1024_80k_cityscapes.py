_base_ = './stdc-1446_512x1024_80k_cityscapes.py'
model = dict(
    backbone=dict(
        stdc_cfg=dict(
            init_cfg=dict(
                checkpoint='./pretrained/pretrained_stdc-1446.pth'))))
