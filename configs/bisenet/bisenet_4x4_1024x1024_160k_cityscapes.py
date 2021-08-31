_base_ = [
    '../_base_/models/bisenet.py',
    '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
