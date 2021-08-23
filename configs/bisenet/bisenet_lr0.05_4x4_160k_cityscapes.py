_base_ = [
    '../_base_/models/bisenet.py',
    '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
