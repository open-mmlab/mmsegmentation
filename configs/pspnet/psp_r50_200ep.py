_base_ = [
    '../_base_/models/psp_r50.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py'
]
cudnn_benchmark = True
# model training and testing settings
train_cfg = dict(sampler=None)
test_cfg = dict(
    crop_scale=713,
    stride=513,
)
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='poly',
    power=0.9,
    by_epoch=False,
)
# runtime settings
total_epochs = 20
evaluation = dict(interval=1, metric='mIoU')
