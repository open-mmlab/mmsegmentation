_base_ = './upernet_vit-b16_mln_512x512_160k_ade20k.py'

model = dict(
    pretrained='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',  # noqa
    backbone=dict(num_heads=6, embed_dims=384, drop_path_rate=0.1),
    decode_head=dict(num_classes=150, in_channels=[384, 384, 384, 384]),
    neck=None,
    auxiliary_head=dict(num_classes=150, in_channels=384))  # yapf: disable
