_base_ = ['./twins_pcpvt-s_fpn_fpnhead_8xb4-80k_ade20k-512x512.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/twins/pcpvt_large_20220308-37579dc6.pth'  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        depths=[3, 8, 27, 3]))
