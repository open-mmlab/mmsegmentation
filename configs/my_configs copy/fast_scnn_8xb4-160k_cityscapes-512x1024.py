_base_ = [
    '../_base_/models/fast_scnn.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
# Re-config the data sampler.
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

# Re-config the optimizer.
optimizer = dict(type='SGD', lr=0.12, momentum=0.9, weight_decay=4e-5)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
