_base_ = ['./mask2former_r50_8xb2-20k_HOTS_v1-512x512.py']
load_from = "checkpoints/mask2former_r101_8xb2-160k_ade20k-512x512_20221203_233905-b7135890.pth"
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
