# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='fixed')
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=70)
checkpoint_config = dict(by_epoch=True, interval=5)
evaluation = dict(interval=10, metric='mIoU', pre_eval=True)
