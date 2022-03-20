_base_ = './icnet_r50-d8_832x832_80k_cityscapes.py'
model = dict(backbone=dict(backbone_cfg=dict(depth=101)))
