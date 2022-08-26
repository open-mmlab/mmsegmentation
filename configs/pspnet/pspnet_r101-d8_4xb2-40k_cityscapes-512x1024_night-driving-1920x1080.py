_base_ = './pspnet_r50-d8_4xb2-40k_cityscapes-512x1024_night-driving-1920x1080.py'  # noqa
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
