_base_ = [
    '../_base_/models/stdc.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
lr_config = dict(warmup='linear', warmup_iters=1000)
train_dataloader = dict(batch_size=12, num_workers=4)
val_dataloader = dict(batch_size=12, num_workers=4)
test_dataloader = val_dataloader
