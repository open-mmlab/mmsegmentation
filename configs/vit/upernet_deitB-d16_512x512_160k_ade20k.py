_base_ = './upernet_vit-d16_512x512_160k_ade20k.py'

model = dict(
    pretrained='https://dl.fbaipublicfiles.com/deit/\
deit_base_distilled_patch16_384-d0272ac0.pth',
    decode_head=dict(num_classes=150),
    auxiliary_head=dict(num_classes=150))
