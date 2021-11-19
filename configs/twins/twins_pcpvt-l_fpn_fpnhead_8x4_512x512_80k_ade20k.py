_base_ = [
    '../_base_/models/twins_pcpvt-s_fpn.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='PCPVT',
        init_cfg=dict(
            type='Pretrained', checkpoint='pretrained/pcpvt_large.pth'),
        depths=[3, 8, 27, 3]))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
