_base_ = './pidnet-s_2xb6-120k_1024x1024-cityscapes.py'

model = dict(
    backbone=dict(
        channels=64,
        init_cfg=dict(checkpoint='pretrain/PIDNet_M_ImageNet.pth')),
    decode_head=dict(in_channels=256))
