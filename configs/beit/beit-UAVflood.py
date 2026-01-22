_base_ = [
    '../_base_/models/upernet_beit.py', '../_base_/datasets/SARflood.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        img_size=(256, 256)),
    decode_head=dict(
        num_classes=2),
    auxiliary_head=dict(
        num_classes=2),
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(170, 170)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=3e-5, betas=(0.9, 0.999), weight_decay=0.05),
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=20000,
        eta_min=0.0,
        by_epoch=False,
    )
]


train_dataloader = dict(batch_size=8, num_workers=8)
val_dataloader = dict(batch_size=8, num_workers=8)
test_dataloader = val_dataloader


