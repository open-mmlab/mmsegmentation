_base_ = [
    '../_base_/models/twins_fpn.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', 'twins_schedule_80k.py'
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
