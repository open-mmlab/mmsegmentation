_base_ = [
    '../_base_/models/bisenetv2.py',
    '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(batch_size=8, num_workers=4)
test_dataloader = val_dataloader
