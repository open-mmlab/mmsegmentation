_base_ = [
    './upernet_swin_base_patch4_window12_512x512_160k_ade20k_'
    'pretrain_384x384_1K.py'
]
model = dict(
    pretrained=\
    'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth', # noqa
)
