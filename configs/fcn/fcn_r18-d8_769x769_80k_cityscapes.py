_base_ = './fcn_r50-d8_769x769_80k_cityscapes.py'
model = dict(
    pretrained='https://download.openmmlab.com/pretrain/third_party/resnet18_v1c-b5776b93.pth',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))