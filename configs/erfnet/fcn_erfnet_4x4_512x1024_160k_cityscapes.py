_base_ = [
    '../_base_/models/fcn_erfnet.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
