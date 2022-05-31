_base_ = [
    '../_base_/models/bisenetv1_r18-d32.py',
    '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.025)
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=4, num_workers=4)
test_dataloader = val_dataloader
