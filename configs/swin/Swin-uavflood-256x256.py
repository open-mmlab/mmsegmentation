_base_ = [
    '../_base_/datasets/SARflood.py',
    '../_base_/default_runtime.py',
    '../_base_/models/upernet_swin.py'
]

# ===== 基础配置 =====
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
crop_size = (256, 256)

# ===== 数据预处理器 =====
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    #mean=[117.926186, 117.568402, 97.217239],
    #std=[53.542876104049824, 50.084170325219176, 50.49331035114637],
    #mean= [432.02181, 315.92948, 246.468659, 310.61462, 360.267789],
    ##std= [97.73313111900238, 85.78646917160748, 95.78015824658593, 124.84677067613467, 251.73965882246978],
    #mean=[0.23651549, 0.31761484, 0.18514981,   0.26901252, -14.57879175,  -8.6098158,  -14.2907338,  -8.33534564],
    #std=[0.16280619, 0.20849304, 0.14008107, 0.19767644, 4.07141682, 3.94773216, 4.21006244, 4.05494136],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255
)

# ===== 模型配置 - Swin-Base =====
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=128,  # Base: 128
        in_channels=8,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],  # Base: [2, 2, 18, 2]
        num_heads=[4, 8, 16, 32],  # Base: [4, 8, 16, 32]
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],  # Base输出通道
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,  # Base stage2输出
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(170, 170))
)

# ===== 优化器配置 =====
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00006,
        betas=(0.9, 0.999),
        weight_decay=0.01
    ),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }
    )
)

# ===== 学习率调度器 =====
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=True,
        begin=0,
        end=5  # warmup前5个epoch
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=5,
        end=100,
        by_epoch=True,
    )
]

# 使用EpochBasedTrainLoop训练100个epoch
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,
    val_interval=10)  # 每10个epoch验证一次

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 修改hooks为基于epoch
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=10,  # 每10个epoch保存一次
        max_keep_ckpts=3,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# 设置日志按epoch显示
log_processor = dict(by_epoch=True)

# ===== 数据加载器配置 =====
train_dataloader = dict(batch_size=8, num_workers=8)  # Base模型较大，减小batch_size
val_dataloader = dict(batch_size=8, num_workers=8)
test_dataloader = val_dataloader

# ===== 随机种子 =====
randomness = dict(seed=42, deterministic=False)
