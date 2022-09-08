_base_ = './deeplabv3plus_r50-d8_321x321_160k_lip.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
