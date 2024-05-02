_base_ = [
    # '../_base_/models/upernet_vit-b16_ln_mln.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
# copied from vit_vit-b16_mln_upernet_8xb2-160k_ade20k-512x512.py

crop_size = (512, 512)
scale = (3000, 3000)
downsample_factor = 5
GT_type='SOD'
num_classes = 6
# dataset settings
dataset_type = 'AI4Arctic'
data_root_train = '/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3'
data_root_test = '/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_test_v3'

gt_root = '/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3_segmaps'
test_root = '/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_test_v3_segmaps'

finetune_ann_file = '/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3/finetune_20.txt'
# finetune_ann_file = '/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3/test1file.txt'
# finetune_ann_file = '/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3/test1file.txt'


test_ann_file = '/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_test_v3/test.txt'
train_pipeline = [
    dict(type='PreLoadImageandSegFromNetCDFFile', data_root=data_root_train, gt_root=gt_root, ann_file=finetune_ann_file, channels=[
        'nersc_sar_primary', 'nersc_sar_secondary'], mean=[-14.508254953309349, -24.701211250236728],
        std=[5.659745919326586, 4.746759336539111], to_float32=True, nan=255, downsample_factor=downsample_factor, with_seg=True, GT_type=GT_type),
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
    dict(type='PreLoadImageandSegFromNetCDFFile', data_root=data_root_test, gt_root=test_root, ann_file=test_ann_file, channels=[
        'nersc_sar_primary', 'nersc_sar_secondary'], mean=[-14.508254953309349, -24.701211250236728],
        std=[5.659745919326586, 4.746759336539111], to_float32=True, nan=255, downsample_factor=downsample_factor, with_seg=False, GT_type=GT_type),
    # dict(type='Resize', scale=scale, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadGTFromPNGFile', gt_root=test_root,
         downsample_factor=downsample_factor, GT_type=GT_type),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='PreLoadImageandSegFromNetCDFFile', data_root=data_root_test, gt_root=test_root, ann_file=test_ann_file, channels=[
         'nersc_sar_primary', 'nersc_sar_secondary'], mean=[-14.508254953309349, -24.701211250236728],
         std=[5.659745919326586, 4.746759336539111], to_float32=True, nan=255, downsample_factor=downsample_factor, with_seg=False, GT_type=GT_type),
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
    mean=[0, 0],
    std=[1, 1],
    bgr_to_rgb=False,
    pad_val=255,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    # pretrained='/project/6075102/AI4arctic/m32patel/mmselfsup/work_dirs/selfsup/mae_vit-base-p16/epoch_200.pth',
    # pretrained=None,
    backbone=dict(
        type='MAE',
        # pretrained='/home/m32patel/projects/def-dclausi/AI4arctic/m32patel/mmselfsup/work_dirs/selfsup/mae_vit-base-p16_cs512-amp-coslr-400e_ai4arctic_norm_pix/epoch_400.pth',
        # pretrained='/home/m32patel/projects/rrg-dclausi/ai4arctic/m32patel/mmsegmentation/work_dirs/mae_ai4arctic_ds5_pt_80_ft_20_mr90/iter_20000.pth',
        init_cfg=dict(type='Pretrained', checkpoint='/home/m32patel/projects/def-y2863che/ai4arctic/m32patel/mmselfsup/work_dirs/selfsup/mae_ai4arctic_ds5_pt_80_ft_20/iter_40000.pth', prefix = 'backbone.'),
        img_size=crop_size,
        patch_size=16,
        in_channels=2,
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
    decode_head=dict(
        type='Mask2FormerHead',
        # in_channels=[256, 512, 1024, 2048],
        in_channels=[768, 768, 768,
                     768],  # input channels of pixel_decoder modules
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(  # DeformableDetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None),
            positional_encoding=dict(  # SinePositionalEncoding
                num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(  # SinePositionalEncoding
            num_feats=128, normalize=True),
        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True)),
            init_cfg=None),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(
                        type='mmdet.CrossEntropyLossCost',
                        weight=5.0,
                        use_sigmoid=True),
                    dict(
                        type='mmdet.DiceCost',
                        weight=5.0,
                        pred_act=True,
                        eps=1.0)
                ]),
            sampler=dict(type='mmdet.MaskPseudoSampler'))),
    # model training and testing settings
    train_cfg=dict(),
    # test_cfg=dict(mode='whole'))  # yapf: disable
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(256, 256)))


val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
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
    type='IterBasedTrainLoop', max_iters=20000, val_interval=1000)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegAI4ArcticVisualizationHook', downsample_factor=downsample_factor, metric='f1', num_classes=6))

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
