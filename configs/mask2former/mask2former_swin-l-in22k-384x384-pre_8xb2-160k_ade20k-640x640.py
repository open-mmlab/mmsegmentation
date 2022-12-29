_base_ = ['./mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-640x640.py']
pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'  # noqa

model = dict(
    backbone=dict(
        embed_dims=192,
        num_heads=[6, 12, 24, 48],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    decode_head=dict(num_queries=100, in_channels=[192, 384, 768, 1536]))
