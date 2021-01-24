_base_ = './pspnet_r50-d8_512x1024_80k_a2d2.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
