_base_ = './icnet_r50-d8_4xb2-20k_HOTS_v1-640x480.py'
load_from = "checkpoints/icnet_r101-d8_832x832_160k_cityscapes_20210926_092350-3a1ebf1a.pth"
model = dict(backbone=dict(backbone_cfg=dict(depth=101)))
