# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='poly',
    power=0.9,
    by_epoch=False,
)
# runtime settings
total_epochs = 220
checkpoint_config = dict(interval=20)
evaluation = dict(interval=20, metric='mIoU')
runner_type = 'epoch'
