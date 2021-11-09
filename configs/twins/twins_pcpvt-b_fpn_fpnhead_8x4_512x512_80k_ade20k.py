_base_ = [
    '../_base_/models/twins_fpn.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', 'twins_schedule_80k.py'
]

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/pcpvt_base.pth',
    backbone=dict(type='PCPVT', depths=[3, 4, 18, 3]),
)

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
