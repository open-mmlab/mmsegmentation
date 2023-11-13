_base_ = [
    '../_base_/datasets/feces.py',
    '../_base_/default_runtime.py',
]

class_weight = [0.7, 0.8, 1.1, 1.2]

checkpoint = '/home/panjm/hepengguang/mmlab_he/mmsegmentation-dev-1.x/checkpoints/ddrnet23-in1kpre_3rdparty-9ca29f62.pth'  # noqa
crop_size = (224, 224)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[0.609, 0.604, 0.578],
    std=[0.195, 0.192, 0.202],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='FAPPM_CONV_nocbam',
        in_channels=3,
        channels=64,
        ppm_channels=96,
        norm_cfg=norm_cfg,
        align_corners=False,
        init_cfg=None),
    decode_head=dict(
        type='DDRHead',
        in_channels=64 * 4,
        channels=128,
        dropout_ratio=0.,
        num_classes=4,
        align_corners=False,
        norm_cfg=norm_cfg,
        loss_decode=[
            dict(
                type='LovaszLoss', class_weight=class_weight, loss_weight=1.0),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=0.4),
        ]),
    auxiliary_head=dict(
        type='DDRHead',
        in_channels=64 * 4,
        channels=128,
        dropout_ratio=0.,
        num_classes=4,
        align_corners=False,
        norm_cfg=norm_cfg,
        loss_decode=[
            dict(
                type='LovaszLoss', class_weight=class_weight, loss_weight=1.0),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=0.4),
        ]),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

iters = 6000
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.0001,
        power=0.9,
        begin=0,
        end=iters,
        by_epoch=False)
]

# training schedule for 120k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=iters, val_interval=100)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=100,
        max_keep_ckpts=2,
        save_best='mDice'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

randomness = dict(seed=304)
