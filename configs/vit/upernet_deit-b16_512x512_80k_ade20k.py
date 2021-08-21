_base_ = './upernet_vit-b16_mln_512x512_80k_ade20k.py'

model = dict(
    pretrained='pretrain/deit.pth',
    backbone=dict(drop_path_rate=0.1),
    neck=None)
