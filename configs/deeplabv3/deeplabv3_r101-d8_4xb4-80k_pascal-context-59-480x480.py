_base_ = './deeplabv3_r50-d8_4xb4-40k_pascal-context-59-480x480.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
