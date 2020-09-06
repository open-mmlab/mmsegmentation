_base_ = './deeplabv3_r50-d16-mg124_512x1024_40k_cityscapes.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
