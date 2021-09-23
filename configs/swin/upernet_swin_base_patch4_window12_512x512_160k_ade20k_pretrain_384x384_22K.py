_base_ = [
    './upernet_swin_base_patch4_window12_512x512_160k_ade20k_'
    'pretrain_384x384_1K.py'
]
model = dict(pretrained='pretrain/swin_base_patch4_window12_384_22k.pth')
