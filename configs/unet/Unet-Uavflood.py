_base_ = [
    '../_base_/models/deeplabv3_unet_s5-d16.py',
    '../_base_/datasets/SARflood.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2),
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(170, 170)))

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=1500,
        end=20000,
        by_epoch=False,
    )
]

randomness = dict(
    seed=42,
    deterministic=False,  # 如需完全可复现，设为True
)

train_dataloader = dict(batch_size=8, num_workers=8)
val_dataloader = dict(batch_size=8, num_workers=8)
test_dataloader = val_dataloader