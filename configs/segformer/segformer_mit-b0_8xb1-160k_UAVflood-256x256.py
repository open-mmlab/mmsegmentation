_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/UAVflood.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=2  # 修改为你需要的类别数
    )
)

randomness = dict(
    seed=42,
    deterministic=False,  # 如需完全可复现，设为True
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=20000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=8, num_workers=8)
val_dataloader = dict(batch_size=8, num_workers=8)
test_dataloader = val_dataloader
