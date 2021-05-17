_base_ = [
    '../_base_/models/setr_pup.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(img_size=(512, 1024), backbone=dict(drop_rate=0.))

optimizer = dict(
    lr=0.01,
    weight_decay=0.0,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}),
)

data = dict(samples_per_gpu=1)
