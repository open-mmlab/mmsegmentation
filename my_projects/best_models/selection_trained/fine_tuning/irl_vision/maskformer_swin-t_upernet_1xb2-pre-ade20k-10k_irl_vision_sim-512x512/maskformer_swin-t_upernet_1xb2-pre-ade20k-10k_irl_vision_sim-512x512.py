backbone_embed_multi = dict(decay_mult=0.0, lr_mult=1.0)
backbone_norm_cfg = dict(requires_grad=True, type='LN')
backbone_norm_multi = dict(decay_mult=0.0, lr_mult=1.0)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'
crop_size = (
    512,
    512,
)
custom_keys = dict({
    'backbone':
    dict(lr_mult=1.0),
    'backbone.norm':
    dict(decay_mult=0.0, lr_mult=1.0),
    'backbone.patch_embed.norm':
    dict(decay_mult=0.0, lr_mult=1.0),
    'backbone.stages.0.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=1.0),
    'backbone.stages.0.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=1.0),
    'backbone.stages.0.downsample.norm':
    dict(decay_mult=0.0, lr_mult=1.0),
    'backbone.stages.1.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=1.0),
    'backbone.stages.1.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=1.0),
    'backbone.stages.1.downsample.norm':
    dict(decay_mult=0.0, lr_mult=1.0),
    'backbone.stages.2.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=1.0),
    'backbone.stages.2.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=1.0),
    'backbone.stages.2.blocks.2.norm':
    dict(decay_mult=0.0, lr_mult=1.0),
    'backbone.stages.2.blocks.3.norm':
    dict(decay_mult=0.0, lr_mult=1.0),
    'backbone.stages.2.blocks.4.norm':
    dict(decay_mult=0.0, lr_mult=1.0),
    'backbone.stages.2.blocks.5.norm':
    dict(decay_mult=0.0, lr_mult=1.0),
    'backbone.stages.2.downsample.norm':
    dict(decay_mult=0.0, lr_mult=1.0),
    'backbone.stages.3.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=1.0),
    'backbone.stages.3.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=1.0),
    'query_embed':
    dict(decay_mult=0.0),
    'relative_position_bias_table':
    dict(decay_mult=0.0, lr_mult=1.0)
})
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
data_root = '/media/ids/Ubuntu files/data/irl_vision_sim/SemanticSegmentation/'
dataset_type = 'IRLVisionSimDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=5000, save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
depths = [
    2,
    2,
    6,
    2,
]
embed_multi = dict(decay_mult=0.0)
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
load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/maskformer/maskformer_swin-t_upernet_8xb2-160k_ade20k-512x512/maskformer_swin-t_upernet_8xb2-160k_ade20k-512x512_20221114_232813-f14e7ce0.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        act_cfg=dict(type='GELU'),
        attn_drop_rate=0.0,
        depths=[
            2,
            2,
            6,
            2,
        ],
        drop_path_rate=0.3,
        drop_rate=0.0,
        embed_dims=96,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth',
            type='Pretrained'),
        mlp_ratio=4,
        norm_cfg=dict(requires_grad=True, type='LN'),
        num_heads=[
            3,
            6,
            12,
            24,
        ],
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_norm=True,
        patch_size=4,
        pretrain_img_size=(
            640,
            480,
        ),
        qk_scale=None,
        qkv_bias=True,
        strides=(
            4,
            2,
            2,
            2,
        ),
        type='SwinTransformer',
        use_abs_pos_embed=False,
        window_size=7),
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
        enforce_decoder_input_project=False,
        feat_channels=256,
        in_channels=[
            96,
            192,
            384,
            768,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_cls=dict(
            class_weight=[
                0.1,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            loss_weight=1.0,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False),
        loss_dice=dict(
            activate=True,
            eps=1.0,
            loss_weight=1.0,
            naive_dice=True,
            reduction='mean',
            type='mmdet.DiceLoss',
            use_sigmoid=True),
        loss_mask=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=20.0,
            reduction='mean',
            type='mmdet.FocalLoss',
            use_sigmoid=True),
        num_classes=72,
        num_queries=100,
        out_channels=256,
        pixel_decoder=dict(
            act_cfg=dict(type='ReLU'),
            norm_cfg=dict(num_groups=32, type='GN'),
            type='mmdet.PixelDecoder'),
        positional_encoding=dict(normalize=True, num_feats=128),
        train_cfg=dict(
            assigner=dict(
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=1.0),
                    dict(
                        binary_input=True,
                        type='mmdet.FocalLossCost',
                        weight=20.0),
                    dict(
                        eps=1.0,
                        pred_act=True,
                        type='mmdet.DiceCost',
                        weight=1.0),
                ],
                type='mmdet.HungarianAssigner'),
            sampler=dict(type='mmdet.MaskPseudoSampler')),
        transformer_decoder=dict(
            init_cfg=None,
            layer_cfg=dict(
                cross_attn_cfg=dict(
                    attn_drop=0.1,
                    batch_first=True,
                    dropout_layer=None,
                    embed_dims=256,
                    num_heads=8,
                    proj_drop=0.1),
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    add_identity=True,
                    dropout_layer=None,
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.1,
                    num_fcs=2),
                self_attn_cfg=dict(
                    attn_drop=0.1,
                    batch_first=True,
                    dropout_layer=None,
                    embed_dims=256,
                    num_heads=8,
                    proj_drop=0.1)),
            num_layers=6,
            return_intermediate=True),
        type='MaskFormerHead'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
num_classes = 72
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=6e-05, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict({
            'backbone':
            dict(lr_mult=1.0),
            'backbone.norm':
            dict(decay_mult=0.0, lr_mult=1.0),
            'backbone.patch_embed.norm':
            dict(decay_mult=0.0, lr_mult=1.0),
            'backbone.stages.0.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=1.0),
            'backbone.stages.0.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=1.0),
            'backbone.stages.0.downsample.norm':
            dict(decay_mult=0.0, lr_mult=1.0),
            'backbone.stages.1.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=1.0),
            'backbone.stages.1.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=1.0),
            'backbone.stages.1.downsample.norm':
            dict(decay_mult=0.0, lr_mult=1.0),
            'backbone.stages.2.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=1.0),
            'backbone.stages.2.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=1.0),
            'backbone.stages.2.blocks.2.norm':
            dict(decay_mult=0.0, lr_mult=1.0),
            'backbone.stages.2.blocks.3.norm':
            dict(decay_mult=0.0, lr_mult=1.0),
            'backbone.stages.2.blocks.4.norm':
            dict(decay_mult=0.0, lr_mult=1.0),
            'backbone.stages.2.blocks.5.norm':
            dict(decay_mult=0.0, lr_mult=1.0),
            'backbone.stages.2.downsample.norm':
            dict(decay_mult=0.0, lr_mult=1.0),
            'backbone.stages.3.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=1.0),
            'backbone.stages.3.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=1.0),
            'query_embed':
            dict(decay_mult=0.0),
            'relative_position_bias_table':
            dict(decay_mult=0.0, lr_mult=1.0)
        })),
    type='OptimWrapper')
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ),
    lr=6e-05,
    momentum=0.9,
    type='AdamW',
    weight_decay=0.01)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=5000, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=5000,
        by_epoch=False,
        end=10000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/test', seg_map_path='ann_dir/test'),
        data_root=
        '/media/ids/Ubuntu files/data/irl_vision_sim/SemanticSegmentation/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='IRLVisionSimDataset'),
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
        data_root=
        '/media/ids/Ubuntu files/data/irl_vision_sim/SemanticSegmentation/',
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
        type='IRLVisionSimDataset'),
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
        data_root=
        '/media/ids/Ubuntu files/data/irl_vision_sim/SemanticSegmentation/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='IRLVisionSimDataset'),
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
work_dir = './work_dirs/maskformer_swin-t_upernet_1xb2-pre-ade20k-10k_irl_vision_sim-512x512'
