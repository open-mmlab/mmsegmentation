_base_ = './ccnet_r50-d8_473x473_160k_lip.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
