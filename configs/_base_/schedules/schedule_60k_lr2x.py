# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=2e-4, by_epoch=False)
# runtime settings
total_iters = 60000
checkpoint_config = dict(by_epoch=False, interval=6000, create_symlink=False)
evaluation = dict(interval=6000, metric='mIoU')
