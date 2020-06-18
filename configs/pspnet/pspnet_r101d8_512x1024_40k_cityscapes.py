_base_ = './pspnet_r50d8_512x1024_40k_cityscapes.py'
model = dict(
    pretrained='pretrain_model/resnet101_v1c_trick-e67eebb6.pth',
    backbone=dict(depth=101))
