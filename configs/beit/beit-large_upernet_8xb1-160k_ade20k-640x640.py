_base_ = [
    '../_base_/models/upernet_beit.py', '../_base_/datasets/ade20k_640x640.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_320k.py'
]
crop_size = (640, 640)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='pretrain/beit_large_patch16_224_pt22k_ft22k.pth',
    backbone=dict(
        type='BEiT',
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4,
        qv_bias=True,
        init_values=1e-6,
        drop_path_rate=0.2,
        out_indices=[7, 11, 15, 23]),
    neck=dict(embed_dim=1024, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        in_channels=[1024, 1024, 1024, 1024], num_classes=150, channels=1024),
    auxiliary_head=dict(in_channels=1024, num_classes=150),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(426, 426)))

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=2e-5, betas=(0.9, 0.999), weight_decay=0.05),
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.95),
    accumulative_counts=2)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=3000),
    dict(
        type='PolyLR',
        power=1.0,
        begin=3000,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=1)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
