_base_ = './pspnet_r50-d8_512x512_40k_voc12aug.py'
model = dict(
    pretrained='pretrain_model/resnet101_v1c_trick-e67eebb6.pth',
    backbone=dict(depth=101))
