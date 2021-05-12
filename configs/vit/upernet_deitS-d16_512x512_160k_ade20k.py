_base_ = './upernet_vit-d16_512x512_160k_ade20k.py'

model = dict(
    pretrained='https://dl.fbaipublicfiles.com/deit/\
deit_small_distilled_patch16_224-649709d9.pth',
    backbone=dict(num_heads=6, embed_dim=384),
    neck=dict(in_channels=[384]),
    decode_head=dict(num_classes=150),
    auxiliary_head=dict(num_classes=150))
