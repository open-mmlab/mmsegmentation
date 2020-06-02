_base_ = './deeplabv3_r50_160ki_ade.py'
model = dict(
    pretrained='pretrain_model/resnet101_v1c_trick-e67eebb6.pth',
    backbone=dict(depth=101))
