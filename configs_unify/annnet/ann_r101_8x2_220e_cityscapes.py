_base_ = './ann_r50_8x2_220e_cityscapes.py'
model = dict(pretrained=None, backbone=dict(depth=101))
