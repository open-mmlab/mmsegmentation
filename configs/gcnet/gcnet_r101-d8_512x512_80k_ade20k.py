_base_ = './gcnet_r50-d8_512x512_80k_ade20k.py'
model = dict(
    pretrained='pretrain_model/resnet101_v1c_trick-e67eebb6.pth',
    backbone=dict(depth=101))
