_base_ = './ann_r50-d8_4xb2-40k_cityscapes-769x769.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
