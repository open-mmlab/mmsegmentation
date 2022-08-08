_base_ = './pspnet_r50-d8_473x473_80k_lip.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
