_base_ = './uper_r50_60ki_cityscapes.py'
model = dict(pretrained=None, backbone=dict(depth=101))
