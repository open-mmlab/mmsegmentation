_base_ = ['./mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-640x640.py']

pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth'  # noqa
model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=pretrained)))
