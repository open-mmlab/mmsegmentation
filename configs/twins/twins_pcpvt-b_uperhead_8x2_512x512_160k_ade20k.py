_base_ = ['./twins_pcpvt-s_uperhead_8x4_512x512_160k_ade20k.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/twins/pcpvt_base_20220308-0621964c.pth'  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        depths=[3, 4, 18, 3],
        drop_path_rate=0.3))

data = dict(samples_per_gpu=2, workers_per_gpu=2)
