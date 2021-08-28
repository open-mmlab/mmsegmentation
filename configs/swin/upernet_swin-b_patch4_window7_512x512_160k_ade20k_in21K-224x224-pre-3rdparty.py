_base_ = [
    'upernet_swin-t_patch4_window7_512x512_160k_ade20k_'
    'in1K-224x224-pre-3rdparty.py'
]
model = dict(pretrained='pretrain/swin_base_patch4_window7_224_22k.pth')
