"""
Sen2GF3Floods fine-tune: fused Sentinel-2 (4ch) + GF-3 (2ch) = 6-band input.

Sentinel-2: pre-disaster optical (R, G, B, NIR)
GF-3: post-disaster SAR (HH, HV)

Expected layout (see tools/setup_sen2gf3floods.py)::

    data/Sen2GF3Floods/Sen2GF3Floods/
        sentinel2/<name>.tif    # 4-band
        gaofen3/<name>.tif      # 2-band
        label/<name>.tif        # 0=bg, 1=flood
        splits/{train,val,test}.txt

Setup (run once):
    python tools/setup_sen2gf3floods.py \
        --data-root data/Sen2GF3Floods/Sen2GF3Floods

Training:
    python tools/train.py configs/floodnet/finetune_sen2gf3floods.py \\
        --work-dir work_dirs/generalization/sen2gf3floods/ \\
        --cfg-options \\
            load_from="work_dirs/floodnet/SwinmoeB/655/best_mIoU_epoch_100.pth"

Testing:
    python tools/test.py configs/floodnet/finetune_sen2gf3floods.py \\
        work_dirs/generalization/sen2gf3floods/best_mIoU_epoch_XX.pth \\
        --work-dir work_dirs/generalization/sen2gf3floods/eval/ \\
        --show-dir vis_pred \\
        --cfg-options \\
            visualizer.alpha=1.0 \\
            test_evaluator.iou_metrics="[mIoU,mDice,mFscore]"
"""

_base_ = ['./finetune_stem_decoder.py']

# ==================== Modal / dataset identifiers ====================
MODAL_NAME = 'sen2gf3'
MODAL_CHANNELS = 6     # 4 (S2 RGBN) + 2 (GF3 HH/HV)

ALL_KNOWN_MODALS = {
    '_delete_': True,
    MODAL_NAME: {
        'channels': MODAL_CHANNELS,
        'pattern': 'sen2gf3',
        'description': 'Sen2GF3Floods fused Sentinel-2 + GF-3 (6-band)',
    },
}
TRAINING_MODALS = [MODAL_NAME]
DATASET_NAMES = [MODAL_NAME]

crop_size = (256, 256)

# ==================== Model override ====================
model = dict(
    dataset_names=DATASET_NAMES,
    backbone=dict(
        modal_configs=ALL_KNOWN_MODALS,
        training_modals=TRAINING_MODALS,
    ),
)

# ==================== Dataset / pipelines ====================
dataset_type = 'Sen2GF3FloodsDataset'
data_root = 'data/Sen2GF3Floods/Sen2GF3Floods/'

train_pipeline = [
    dict(type='LoadSen2GF3FloodsImage', to_float32=True),
    dict(type='LoadSen2GF3FloodsAnnotation'),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='MultiModalNormalize'),
    dict(type='MultiModalPad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='PackMultiModalSegInputs',
         meta_keys=('img_path', 'ori_filename', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'modal_type', 'actual_channels', 'dataset_name',
                    'img_norm_cfg', 'reduce_zero_label')),
]

test_pipeline = [
    dict(type='LoadSen2GF3FloodsImage', to_float32=True),
    dict(type='LoadSen2GF3FloodsAnnotation'),
    dict(type='MultiModalNormalize'),
    dict(type='PackMultiModalSegInputs',
         meta_keys=('img_path', 'ori_filename', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'modal_type', 'actual_channels', 'dataset_name',
                    'img_norm_cfg', 'reduce_zero_label')),
]

# ==================== Dataloaders ====================
train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(
        _delete_=True,
        type='DefaultSampler',
        shuffle=True,
    ),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='splits/train.txt',
        data_prefix=dict(
            img_path='sentinel2',
            seg_map_path='label',
        ),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='splits/val.txt',
        data_prefix=dict(
            img_path='sentinel2',
            seg_map_path='label',
        ),
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='splits/test.txt',
        data_prefix=dict(
            img_path='sentinel2',
            seg_map_path='label',
        ),
        pipeline=test_pipeline,
    ),
)
