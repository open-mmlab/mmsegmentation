_base_ = './vit_vit-b16_mln_upernet_8xb2-160k_ade20k-512x512.py'

model = dict(
    pretrained='pretrain/deit_base_patch16_224-b5f2ef4d.pth',
    backbone=dict(drop_path_rate=0.1, final_norm=True))
