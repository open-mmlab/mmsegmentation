_base_ = [
    '../../../../configs/_base_/default_runtime.py'
]

# ============================================================
# Registry
# ============================================================
# EncoderDecoder、UNet、ASPPHeadなどをmmsegのRegistryから
# 確実に取得できるようにする。
default_scope = 'mmseg'

custom_imports = dict(
    imports=['mmseg.models'],
    allow_failed_imports=False,
)

# ============================================================
# Dataset
# ============================================================
dataset_type = 'BaseSegDataset'

# 0 = 背景
# 1 = 確信度の高いフィラメント
# 2 = 候補フィラメント
# 255 = IGNORE
metainfo = dict(
    classes=(
        'background',
        'confirmed_filament',
        'candidate_filament',
    ),
    palette=[
        [0, 0, 0],        # background
        [255, 220, 0],    # confirmed_filament
        [0, 150, 255],    # candidate_filament
    ],
)

# ============================================================
# Data root
# ============================================================
# 元のパスから変更しない
data_root = '/mnt/sp4t/fujiwara.r/split'

# ============================================================
# Data Preprocessor
# ============================================================
data_preprocessor = dict(
    type='SegDataPreProcessor',

    # PNGは0～255のuint8。
    # ImageNet事前学習を使用しないため、127.5で中心化する。
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],

    bgr_to_rgb=True,

    # パッチサイズ
    size=(256, 256),

    pad_val=0,
    seg_pad_val=255,
)

# ============================================================
# Model
# ============================================================
# UNet、ASPPHead、FCNHeadの構成は変更しない。
model = dict(
    type='EncoderDecoder',

    data_preprocessor=data_preprocessor,

    pretrained=None,

    # --------------------------------------------------------
    # Backbone
    # --------------------------------------------------------
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=5,

        strides=(1, 1, 1, 1, 1),

        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),

        downsamples=(True, True, True, True),

        norm_cfg=dict(
            type='BN',
            requires_grad=True,
        ),

        act_cfg=dict(
            type='ReLU',
        ),

        upsample_cfg=dict(
            type='InterpConv',
        ),
    ),

    # --------------------------------------------------------
    # Main decode head
    # --------------------------------------------------------
    decode_head=dict(
        type='ASPPHead',

        in_channels=64,
        in_index=4,

        channels=16,

        dilations=(1, 12, 24, 36),

        dropout_ratio=0.1,

        num_classes=3,

        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                loss_weight=1.0,

                # 背景約89.8%、Class 1約3%、Class 2約7%を
                # 想定したクラス重み。
                class_weight=[0.2, 3.0, 1.5],
            ),
            dict(
                type='DiceLoss',
                loss_name='loss_dice',
                loss_weight=3.0,
            ),
        ],
    ),

    # --------------------------------------------------------
    # Auxiliary head
    # --------------------------------------------------------
    auxiliary_head=dict(
        type='FCNHead',

        in_channels=128,
        in_index=3,

        channels=64,

        num_classes=3,
        num_convs=1,

        loss_decode=dict(
            type='CrossEntropyLoss',
            loss_weight=0.4,
        ),
    ),

    # モデル内部のtrain_cfgには学習ループを書かない。
    # IterBasedTrainLoopは後述のトップレベルtrain_cfgで設定する。
    train_cfg=dict(),

    test_cfg=dict(
        mode='whole',
    ),
)

# ============================================================
# Pipeline
# ============================================================
train_pipeline = [
    dict(
        type='LoadImageFromFile',
    ),
    dict(
        type='LoadAnnotations',
    ),
    dict(
        type='RandomFlip',
        direction=['horizontal', 'vertical'],
        prob=0.5,
    ),
    dict(
        type='PackSegInputs',
    ),
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
    ),
    dict(
        type='Resize',
        scale=(256, 256),
        keep_ratio=True,
    ),
    dict(
        type='LoadAnnotations',
    ),
    dict(
        type='PackSegInputs',
    ),
]

# ============================================================
# Train Dataloader
# ============================================================
train_dataloader = dict(
    batch_size=8,

    num_workers=4,
    persistent_workers=True,

    sampler=dict(
        type='InfiniteSampler',
        shuffle=True,
    ),

    dataset=dict(
        type=dataset_type,

        metainfo=metainfo,

        data_root=data_root,

        data_prefix=dict(
            img_path='train/in',
            seg_map_path='train/out',
        ),

        img_suffix='_in.png',
        seg_map_suffix='_out.png',

        pipeline=train_pipeline,
    ),
)

# ============================================================
# Validation Dataloader
# ============================================================
val_dataloader = dict(
    batch_size=1,

    num_workers=4,
    persistent_workers=True,

    sampler=dict(
        type='DefaultSampler',
        shuffle=False,
    ),

    dataset=dict(
        type=dataset_type,

        metainfo=metainfo,

        data_root=data_root,

        data_prefix=dict(
            img_path='val/in',
            seg_map_path='val/out',
        ),

        img_suffix='_in.png',
        seg_map_suffix='_out.png',

        pipeline=test_pipeline,
    ),
)

# ============================================================
# Test Dataloader
# ============================================================
test_dataloader = dict(
    batch_size=1,

    num_workers=4,
    persistent_workers=True,

    sampler=dict(
        type='DefaultSampler',
        shuffle=False,
    ),

    dataset=dict(
        type=dataset_type,

        metainfo=metainfo,

        data_root=data_root,

        data_prefix=dict(
            img_path='test/in',
            seg_map_path='test/out',
        ),

        img_suffix='_in.png',
        seg_map_suffix='_out.png',

        pipeline=test_pipeline,
    ),
)

# 分割先のtrain/val/test構成は変更しない。
# 元の領域別分割結果をそのままプールして読み込む。
# :contentReference[oaicite:1]{index=1}

# ============================================================
# Evaluator
# ============================================================
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=[
        'mDice',
        'mIoU',
    ],
)

test_evaluator = val_evaluator

# ============================================================
# Train / Validation / Test Loop
# ============================================================
train_cfg = dict(
    type='IterBasedTrainLoop',

    # 学習量は元の設定から変更しない
    max_iters=8000,

    # 500回ごとから1000回ごとへ変更。
    # Validation実行回数を減らして総時間を短縮する。
    val_interval=1000,
)

val_cfg = dict(
    type='ValLoop',
)

test_cfg = dict(
    type='TestLoop',
)

# ============================================================
# Hooks
# ============================================================
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',

        by_epoch=False,

        # チェックポイントは安全のため500回ごとに残す。
        # Validationより頻繁でも問題ない。
        interval=500,

        # Validation実行時にmDiceが最高ならベストモデルを保存する。
        save_best='mDice',

        rule='greater',

        # 保存容量を抑える場合に使用。
        # 最新3件の通常チェックポイントを保持する。
        max_keep_ckpts=3,
    ),
)

# ============================================================
# Optimizer
# ============================================================
# 元のSGD、学習率、momentum、weight decayは変更しない。
#
# OptimWrapperからAmpOptimWrapperへ変更して
# Automatic Mixed Precisionを有効にする。
optim_wrapper = dict(
    type='AmpOptimWrapper',

    # オーバーフローを検出しながらloss scaleを自動調整する。
    loss_scale='dynamic',

    optimizer=dict(
        type='SGD',

        lr=0.01,
        momentum=0.9,
        weight_decay=0.0005,
    ),
)

# ============================================================
# Learning-rate Scheduler
# ============================================================
# max_iters=8000のままなのでendも8000から変更しない。
param_scheduler = [
    dict(
        type='PolyLR',

        begin=0,
        end=8000,

        by_epoch=False,

        power=0.9,
        eta_min=0.0001,
    ),
]

# ============================================================
# Runtime
# ============================================================
randomness = dict(
    seed=0,
)

env_cfg = dict(
    # 入力サイズが256×256で固定されているため有効。
    cudnn_benchmark=True,
)

# 元のwork_dirから変更しない。
work_dir = 'projects/filament_segmentation/transition/work_dirs/unet_deeplabv3_hgbs_3class'


