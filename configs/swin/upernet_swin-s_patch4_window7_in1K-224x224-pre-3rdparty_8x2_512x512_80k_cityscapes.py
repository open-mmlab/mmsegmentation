_base_ = [
    'upernet_swin-t_patch4_window7_in1K-224x224-pre-3rdparty_8x2_'
    '512x1024_80k_cityscapes.py'
]
model = dict(
    pretrained='pretrain/swin_small_patch4_window7_224.pth',
    backbone=dict(depths=[2, 2, 18, 2]),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=150),
    auxiliary_head=dict(in_channels=384, num_classes=150))
