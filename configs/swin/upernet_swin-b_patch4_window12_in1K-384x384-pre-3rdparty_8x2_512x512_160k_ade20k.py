_base_ = [
    'upernet_swin-t_patch4_window7_in1K-224x224-pre-3rdparty_8x2_'
    '512x512_160k_ade20k.py'
]
model = dict(
    pretrained='pretrain/swin_base_patch4_window12_384.pth',
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12),
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=150),
    auxiliary_head=dict(in_channels=512, num_classes=150))
