_base_ = './icnet_r50-d8_832x832_160k_cityscapes.py'
model = dict(backbone=dict(backbone_cfg=dict(depth=101)))
