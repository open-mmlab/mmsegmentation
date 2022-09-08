_base_ = './deeplabv3_r50-d8_321x321_80k_lip.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
