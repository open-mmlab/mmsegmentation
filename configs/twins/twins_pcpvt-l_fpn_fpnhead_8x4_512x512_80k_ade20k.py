_base_ = ['./twins_pcpvt-s_fpn_fpnhead_8x4_512x512_80k_ade20k.py']

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint='pretrained/pcpvt_large.pth'),
        depths=[3, 8, 27, 3]))
