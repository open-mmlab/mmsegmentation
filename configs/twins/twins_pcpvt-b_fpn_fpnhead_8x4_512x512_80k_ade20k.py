_base_ = ['./twins_pcpvt-s_fpn_fpnhead_8x4_512x512_80k_ade20k.py']

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint='pretrained/pcpvt_base.pth'),
        depths=[3, 4, 18, 3]), )
