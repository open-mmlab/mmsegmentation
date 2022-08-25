_base_ = './vit_vit-b16_mln_upernet_8xb2-160k_ade20k-512x512.py'

model = dict(
    pretrained='pretrain/deit_small_patch16_224-cd65a155.pth',
    backbone=dict(num_heads=6, embed_dims=384, drop_path_rate=0.1),
    decode_head=dict(num_classes=150, in_channels=[384, 384, 384, 384]),
    neck=dict(in_channels=[384, 384, 384, 384], out_channels=384),
    auxiliary_head=dict(num_classes=150, in_channels=384))
