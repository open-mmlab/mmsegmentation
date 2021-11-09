_base_ = [
    '../_base_/schedules/schedule_160k.py'
]
# optimizer
optimizer = dict(_delete_=True, type='SGD', lr=0.01, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=16000, metric='mIoU')

find_unused_parameters = True
