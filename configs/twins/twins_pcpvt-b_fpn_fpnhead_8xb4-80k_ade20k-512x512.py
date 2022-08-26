_base_ = ['./twins_pcpvt-s_fpn_fpnhead_8xb4-80k_ade20k-512x512.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/twins/pcpvt_base_20220308-0621964c.pth'  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        depths=[3, 4, 18, 3]), )
