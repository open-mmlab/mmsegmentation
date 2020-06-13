_base_ = './deeplabv3_r50_512x512_80ki_voc12aug.py'
model = dict(
    pretrained='pretrain_model/resnet101_v1c_trick-e67eebb6.pth',
    backbone=dict(depth=101))
