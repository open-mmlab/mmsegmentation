checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/stdc/stdc2_20220308-7dbd9127.pth'  # noqa
_base_ = './stdc2_512x1024_80k_cityscapes.py'
model = dict(
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint))))
