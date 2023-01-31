img_scale = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(512, 512))
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512)),
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'BaseSegDataset'
data_root = 'data/Bactteria_Det'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BaseSegDataset',
        data_root='data/Bactteria_Det',
        img_suffix='.png',
        seg_map_suffix='.png',
        metainfo=dict(classes=('background', 'foreground')),
        data_prefix=dict(
            img_path='images/train/', seg_map_path='masks/train/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', scale=(512, 512), keep_ratio=False),
            dict(type='RandomCrop', crop_size=(512, 512)),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs')
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseSegDataset',
        data_root='data/Bactteria_Det',
        img_suffix='.png',
        seg_map_suffix='.png',
        metainfo=dict(classes=('background', 'foreground')),
        data_prefix=dict(img_path='images/val/', seg_map_path='masks/val/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(512, 512), keep_ratio=False),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseSegDataset',
        data_root='data/Bactteria_Det',
        img_suffix='.png',
        seg_map_suffix='.png',
        metainfo=dict(classes=('background', 'foreground')),
        data_prefix=dict(img_path='images/val/', seg_map_path='masks/val/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(512, 512), keep_ratio=False),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = (None, )
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=None,
    name='visualizer',
    _scope_='mmseg')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
tta_model = dict(type='SegTTAModel', _scope_='mmseg')
optimizer = dict(
    type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005, _scope_='mmseg')
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005),
    clip_grad=None,
    _scope_='mmseg')
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.0001,
        power=0.9,
        begin=0,
        end=20000,
        by_epoch=False,
        _scope_='mmseg')
]
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=20000,
    val_interval=1000,
    _scope_='mmseg')
val_cfg = dict(type='ValLoop', _scope_='mmseg')
test_cfg = dict(type='TestLoop', _scope_='mmseg')
default_hooks = dict(
    timer=dict(type='IterTimerHook', _scope_='mmseg'),
    logger=dict(
        type='LoggerHook',
        interval=50,
        log_metric_by_epoch=False,
        _scope_='mmseg'),
    param_scheduler=dict(type='ParamSchedulerHook', _scope_='mmseg'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=20000,
        _scope_='mmseg'),
    sampler_seed=dict(type='DistSamplerSeedHook', _scope_='mmseg'),
    visualization=dict(type='SegVisualizationHook', _scope_='mmseg'))
work_dir = 'projects/Bactteria_Dataset/work_dirs/Bactteria_Det_unet_\
    0.001_CrossEntropyLoss/'
