_base_ = './upernet_vit-d16_512x512_160k_ade20k.py'

model = dict(
    pretrained='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',  # noqa
    backbone=dict(drop_path_rate=0.1),
    decode_head=dict(num_classes=150),
    auxiliary_head=dict(num_classes=150))  # yapf: disable
