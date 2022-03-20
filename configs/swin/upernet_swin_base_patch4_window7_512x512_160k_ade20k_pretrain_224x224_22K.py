_base_ = [
    './upernet_swin_base_patch4_window7_512x512_160k_ade20k_'
    'pretrain_224x224_1K.py'
]
model = dict(pretrained='pretrain/swin_base_patch4_window7_224_22k.pth')
