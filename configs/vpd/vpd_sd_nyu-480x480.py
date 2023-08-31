_base_ = [
    '../_base_/models/vpd_sd.py', '../_base_/datasets/nyu.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

find_unused_parameters = True

crop_size = (480, 480)

train_dataloader = dict(
    batch_size=1,
    num_workers=8,
)

model = dict(
    type='DepthEstimator',
    data_preprocessor=dict(size=crop_size),
    backbone=dict(
        class_embed_path='vpd/nyu_class_embeddings.pth',
        text_adapter='TextAdapterDepth',
        pad_shape=512,
        unet_cfg=dict(use_attn=False),
    ),
    decode_head=dict(
        type='VPDDepthHead',
        max_depth=10,
        fmap_border=(1, 1),
        feature_dim=1536,
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide_flip', crop_size=(480, 480), stride=(161, 161)))

# optimizer
optim_wrapper = dict(
    optimizer=dict(_delete_=True, type='AdamW', lr=0.001, weight_decay=0.1),
    paramwise_cfg=dict(custom_keys={
        'backbone.unet': dict(lr_mult=0.01),
    }),
    clip_grad=None)

# learning policy
param_scheduler = [
    dict(type='LinearLR', begin=1, end=10000, by_epoch=False),
    dict(
        type='PolyLR',
        eta_min=0.001,
        power=0.9,
        begin=20000,
        end=40000,
        by_epoch=False)
]
