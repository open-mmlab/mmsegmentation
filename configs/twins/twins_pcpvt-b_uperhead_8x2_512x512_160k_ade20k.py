_base_ = ['./twins_pcpvt-s_uperhead_8x4_512x512_160k_ade20k.py']

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint='pretrained/pcpvt_base.pth'),
        depths=[3, 4, 18, 3],
        drop_path_rate=0.3))

data = dict(samples_per_gpu=2, workers_per_gpu=2)
