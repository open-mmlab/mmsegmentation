_base_ = './deeplabv3plus_r50-d8_4xb2-40k_cityscapes-512x1024.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
