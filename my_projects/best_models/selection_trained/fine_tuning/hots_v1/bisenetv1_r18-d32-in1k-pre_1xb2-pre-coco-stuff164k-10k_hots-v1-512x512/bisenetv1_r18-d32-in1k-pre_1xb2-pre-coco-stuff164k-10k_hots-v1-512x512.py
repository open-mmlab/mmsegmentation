crop_size = (
    512,
    512,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = '/media/ids/Ubuntu files/data/HOTS_v1/SemanticSegmentation/'
dataset_type = 'HOTSDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=1000, save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
launcher = 'none'
load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r18-d32_in1k-pre_lr5e-3_4x4_512x512_160k_coco-stuff164k/bisenetv1_r18-d32_in1k-pre_lr5e-3_4x4_512x512_160k_coco-stuff164k_20211023_013100-f700dbf7.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=[
        dict(
            align_corners=False,
            channels=64,
            concat_input=False,
            in_channels=128,
            in_index=1,
            loss_decode=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            num_classes=47,
            num_convs=1,
            type='FCNHead'),
        dict(
            align_corners=False,
            channels=64,
            concat_input=False,
            in_channels=128,
            in_index=2,
            loss_decode=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            num_classes=47,
            num_convs=1,
            type='FCNHead'),
    ],
    backbone=dict(
        align_corners=False,
        backbone_cfg=dict(
            contract_dilation=True,
            depth=18,
            dilations=(
                1,
                1,
                1,
                1,
            ),
            in_channels=3,
            init_cfg=dict(
                checkpoint='open-mmlab://resnet18_v1c', type='Pretrained'),
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            norm_eval=False,
            num_stages=4,
            out_indices=(
                0,
                1,
                2,
                3,
            ),
            strides=(
                1,
                2,
                2,
                2,
            ),
            style='pytorch',
            type='ResNet'),
        context_channels=(
            128,
            256,
            512,
        ),
        in_channels=3,
        init_cfg=None,
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        out_channels=256,
        out_indices=(
            0,
            1,
            2,
        ),
        spatial_channels=(
            64,
            64,
            64,
            128,
        ),
        type='BiSeNetV1'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=256,
        in_index=0,
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=47,
        num_convs=1,
        type='FCNHead'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.005, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(lr=0.005, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(begin=0, by_epoch=False, end=1500, start_factor=0.1, type='LinearLR'),
    dict(
        begin=1500,
        by_epoch=False,
        end=10000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/test', seg_map_path='ann_dir/test'),
        data_root='/media/ids/Ubuntu files/data/HOTS_v1/SemanticSegmentation/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='HOTSDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type="CustomIoUMetric")
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2048,
        512,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=10000, type='IterBasedTrainLoop', val_interval=1000)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        data_root='/media/ids/Ubuntu files/data/HOTS_v1/SemanticSegmentation/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    2048,
                    512,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    512,
                    512,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackSegInputs'),
        ],
        type='HOTSDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            2048,
            512,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        512,
        512,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/eval', seg_map_path='ann_dir/eval'),
        data_root='/media/ids/Ubuntu files/data/HOTS_v1/SemanticSegmentation/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='HOTSDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2048,
        512,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/bisenetv1_r18-d32-in1k-pre_1xb2-pre-coco-stuff164k-10k_hots-v1-512x512'
