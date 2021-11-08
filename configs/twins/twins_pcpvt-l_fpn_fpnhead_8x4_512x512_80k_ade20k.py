_base_ = [
    '../_base_/models/twins_fpn.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', 'twins_schedule_80k.py'
]

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/pcpvt_large.pth',
    backbone=dict(type='Twins_pcpvt', depths=[3, 8, 27, 3]))

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
