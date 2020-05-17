_base_ = './ann_r50_60ki_cityscapes.py'
model = dict(pretrained=None, backbone=dict(depth=101))
