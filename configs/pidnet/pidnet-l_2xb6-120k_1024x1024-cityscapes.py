_base_ = './pidnet-s_2xb6-120k_1024x1024-cityscapes.py'

model = dict(
    backbone=dict(
        channels=64,
        ppm_channels=112,
        num_stem_blocks=3,
        num_branch_blocks=4,
        init_cfg=dict(checkpoint='pretrain/PIDNet_L_ImageNet.pth')),
    decode_head=dict(in_channels=256, channels=256))
