_base_ = './pidnet-s_2xb8-90k_640x640-coco-stuff10k.py'

model = dict(
    backbone=dict(
        channels=64,
        init_cfg=dict(checkpoint='pretrain/PIDNet_M_ImageNet.pth')),
    decode_head=dict(in_channels=256))
