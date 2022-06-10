_base_ = [
    '../_base_/models/lraspp_m-v3-d8.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 1024)
preprocess_cfg = dict(size=crop_size)
# Re-config the data sampler.
model = dict(preprocess_cfg=preprocess_cfg)
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=4, num_workers=4)
test_dataloader = val_dataloader

runner = dict(type='IterBasedRunner', max_iters=320000)
