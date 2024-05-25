_base_ = [
    # '../_base_/models/upernet_vit-b16_ln_mln.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
# copied from vit_vit-b16_mln_upernet_8xb2-160k_ade20k-512x512.py

crop_size = (512, 512)
scale = (3000, 3000)
downsample_factor = 5
GT_type = ['SIC', 'SOD', 'FLOE']
num_classes = {'SIC': 12, 'SOD': 7, 'FLOE': 8} # add 1 class extra for visualization to work correctly, put [11,6,7] in other places
metrics = {'SIC': 'r2', 'SOD': 'f1', 'FLOE': 'f1'}
combined_score_weights = [2, 2, 1]
channels = [
    # -- Sentinel-1 variables -- #
    'nersc_sar_primary',
    'nersc_sar_secondary',
    # 'sar_incidenceangle',

    # -- Geographical variables -- #
    'distance_map',

    # # -- AMSR2 channels -- #
    # 'btemp_6_9h', 'btemp_6_9v',
    # 'btemp_7_3h', 'btemp_7_3v',
    # 'btemp_10_7h', 'btemp_10_7v',
    # 'btemp_18_7h', 'btemp_18_7v',
    # 'btemp_23_8h', 'btemp_23_8v',
    # 'btemp_36_5h', 'btemp_36_5v',
    # 'btemp_89_0h', 'btemp_89_0v',

    # # -- Environmental variables -- #
    # 'u10m_rotated', 'v10m_rotated',
    # 't2m', 'skt', 'tcwv', 'tclw',
]


# GT_type=['SIC']

# dataset settings
dataset_type = 'AI4Arctic'
data_root_train = '/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3'
data_root_test = '/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_test_v3'

gt_root = '/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3_segmaps'
test_root = '/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_test_v3_segmaps'

finetune_ann_file = '/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3/test1file.txt'
test_ann_file = '/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_test_v3/test.txt'
# test_ann_file = '/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3/test1file.txt'

train_pipeline = [
    dict(type='PreLoadImageandSegFromNetCDFFile', data_root=data_root_train, gt_root=gt_root, ann_file=finetune_ann_file, channels=
        channels, mean=[-14.508254953309349, -24.701211250236728, 0],
        std=[5.659745919326586, 4.746759336539111, 1], to_float32=True, nan=255, downsample_factor=downsample_factor, with_seg=True, GT_type=GT_type),
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=scale,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.9),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion')
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='PreLoadImageandSegFromNetCDFFile', data_root=data_root_test, gt_root=test_root, ann_file=test_ann_file, channels=channels, mean=[-14.508254953309349, -24.701211250236728, 0],
        std=[5.659745919326586, 4.746759336539111, 1], to_float32=True, nan=255, downsample_factor=downsample_factor, with_seg=False, GT_type=GT_type),
    # dict(type='Resize', scale=scale, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadGTFromPNGFile', gt_root=test_root,
         downsample_factor=downsample_factor, GT_type=GT_type),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='PreLoadImageandSegFromNetCDFFile', data_root=data_root_test, gt_root=test_root, ann_file=test_ann_file, channels=channels, mean=[-14.508254953309349, -24.701211250236728, 0],
         std=[5.659745919326586, 4.746759336539111, 1], to_float32=True, nan=255, downsample_factor=downsample_factor, with_seg=False, GT_type=GT_type),
    # dict(type='Resize', scale=scale, keep_ratio=True),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ],
            [dict(type='LoadGTFromPNGFile', gt_root=gt_root,
                  downsample_factor=downsample_factor, GT_type=GT_type)],
            [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_train,
        ann_file=finetune_ann_file,
        data_prefix=dict(
            img_path='', seg_map_path=''),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_test,
        ann_file=test_ann_file,
        data_prefix=dict(
            img_path='',
            seg_map_path=''),
        pipeline=test_pipeline))
test_dataloader = val_dataloader


# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[0, 0, 0],
    std=[1, 1, 1],
    bgr_to_rgb=False,
    pad_val=255,
    seg_pad_val=255)
model = dict(
    type='MultitaskEncoderDecoder',
    data_preprocessor=data_preprocessor,
    # pretrained='/project/6075102/AI4arctic/m32patel/mmselfsup/work_dirs/selfsup/mae_vit-base-p16/epoch_200.pth',
    # pretrained=None,
    backbone=dict(
        type='MAE',
        # pretrained='/home/m32patel/projects/def-dclausi/AI4arctic/m32patel/mmselfsup/work_dirs/selfsup/mae_vit-base-p16_cs512-amp-coslr-400e_ai4arctic_norm_pix/epoch_400.pth',
        # pretrained='/project/6075102/AI4arctic/m32patel/mmselfsup/work_dirs/selfsup/mae_vit-base-p16/epoch_200.pth',
        # init_cfg=dict(type='Pretrained', checkpoint=None, prefix = 'backbone.'),
        img_size=crop_size,
        patch_size=16,
        in_channels=len(channels),
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=(3, 5, 7, 11),
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        init_values=0.1),
    neck=dict(type='Feature2Pyramid', embed_dim=768, rescales=[4, 2, 1, 0.5]),
    # decode_head=dict(
    #     type='UPerHead',
    #     in_channels=[768, 768, 768, 768],
    #     in_index=[0, 1, 2, 3],
    #     pool_scales=(1, 2, 3, 6),
    #     channels=768,
    #     dropout_ratio=0.1,
    #     num_classes=6,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    decode_head=[
        dict(
            type='UPerHead',
            task='SIC',
            in_channels=[768, 768, 768, 768],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=768,
            dropout_ratio=0.1,
            num_classes=11,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='UPerHead',
            task='SOD',
            in_channels=[768, 768, 768, 768],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=768,
            dropout_ratio=0.1,
            num_classes=6,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='UPerHead',
            task='FLOE',
            in_channels=[768, 768, 768, 768],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=768,
            dropout_ratio=0.1,
            num_classes=7,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ],
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    # test_cfg=dict(mode='whole'))  # yapf: disable
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(256, 256)))


val_evaluator = dict(type='MultitaskIoUMetric',
                     tasks=GT_type, iou_metrics=['mIoU', 'mFscore'], num_classes=num_classes)
test_evaluator = val_evaluator

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=2000),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=2000,
        end=20000,
        by_epoch=False,
    )
]
# training schedule for 160k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=20, val_interval=10)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False,
                    interval=10, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegAI4ArcticVisualizationHook', tasks=GT_type, num_classes=num_classes, downsample_factor=downsample_factor, metrics=metrics, combined_score_weights=combined_score_weights, draw=True))


GT_type = ['SIC', 'SOD', 'FLOE']
num_classes = {'SIC': 12, 'SOD': 7, 'FLOE': 8}
metrics = {'SIC': 'r2', 'SOD': 'f1', 'FLOE': 'f1'}


vis_backends = [dict(type='WandbVisBackend',
                     init_kwargs=dict(
                         entity='mmwhale',
                         project='MAE-finetune',
                         name='{{fileBasenameNoExtension}}',),
                     #  name='filename',),
                     define_metric_cfg=None,
                     commit=True,
                     log_code_name=None,
                     watch_kwargs=None),
                dict(type='LocalVisBackend')]

visualizer = dict(
    vis_backends=vis_backends)


custom_imports = dict(
    imports=['mmseg.datasets.ai4arctic',
             'mmseg.datasets.transforms.loading_ai4arctic',
             'mmseg.models.segmentors.mutitask_encoder_decoder',
             'mmseg.evaluation.metrics.multitask_iou_metric',
             'mmseg.engine.hooks.ai4arctic_visualization_hook'],
    allow_failed_imports=False)

# custom_imports = dict(
#     imports=[
#              'mmseg.datasets.ai4arctic'],
#     allow_failed_imports=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
# train_dataloader = dict(batch_size=2)
# val_dataloader = dict(batch_size=1)
# test_dataloader = val_dataloader
