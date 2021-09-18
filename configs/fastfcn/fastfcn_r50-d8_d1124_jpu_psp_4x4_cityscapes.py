_base_ = [
    '../_base_/models/fastfcn_r50-d32.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(backbone=dict(
    dilations=(1, 1, 2, 4),
    strides=(1, 2, 1, 1),
))
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
