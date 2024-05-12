_base_ = './fcn-d6_r50-d16_4xb2-20k_HOTS_v1-640x480.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
