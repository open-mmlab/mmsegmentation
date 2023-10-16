_base_ = './uniformer_fpn_light_xs_1xb2-160k_ade20k-512x512.py'

# model settings
model = dict(
    backbone=dict(
        depth=[2, 5, 8, 2],
        prune_ratio=[[], [], [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                     [0.5, 0.5]],
        trade_off=[[], [], [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5]],
        embed_dim=[56, 112, 224, 448],
        head_dim=28,
        drop_path_rate=0.),
    neck=dict(in_channels=[56, 112, 224, 448]),
)

optimizer = dict(type='AdamW', lr=0.0001 * 1, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-8)

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00001, betas=(0.9, 0.999), weight_decay=0.0001))

# learning policy
param_scheduler = [
    # Use a linear warm-up at [0, 100) iterations
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=500),
    # Use a cosine learning rate at [100, 900) iterations
    dict(
        type='CosineAnnealingLR',
        T_max=159500,
        by_epoch=False,
        begin=500,
        end=160000),
]

load_from = '/root/mmsegmentation/fpn_xxs.pth'
