_base_ = './psp_r50_tv_200ep.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
