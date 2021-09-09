_base_ = [
    'upernet_swin-b_patch4_window7_in1K-224x224-pre-3rdparty_8x2_'
    '512x512_160k_ade20k.py'
]
model = dict(pretrained='pretrain/swin_base_patch4_window7_224_22k.pth')
