_base_ = './icnet_r50-d8_4xb2-80k_cityscapes-832x832.py'
model = dict(backbone=dict(backbone_cfg=dict(depth=101)))
