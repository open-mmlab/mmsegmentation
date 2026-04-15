"""
Sen1Floods11 fine-tune: S2Hand (13-band Sentinel-2 MSI).

Sibling of ``finetune_sen1floods11_s1.py`` - same freeze / LR / schedule
strategy, but swaps the trainable modal to ``s2`` (13 ch) and points
the dataset at the ``S2Hand`` subdirectory. LabelHand files are shared
between S1 and S2, so the exact same splits/train.txt / splits/val.txt
/ splits/test.txt files are used - this means any S1 vs S2 comparison
is done on the same tiles.

Expected on-disk layout (see tools/setup_sen1floods11.py)::

    data/Sen1Floods11/
        S1Hand/<base>_S1Hand.tif       # 2-band SAR (unused here)
        S2Hand/<base>_S2Hand.tif       # 13-band Sentinel-2 MSI
        LabelHand/<base>_LabelHand.tif # -1=nodata, 0=bg, 1=flood
        splits/{train,val,test}.txt

All tiles are 512x512.

Setup (run once, before the first training run):

    # 1. Generate splits + compute mean/std (covers both s1 and s2).
    python tools/setup_sen1floods11.py --data-root data/Sen1Floods11

    # 2. Paste the printed NORM_CONFIGS block into the 's2' entry in
    #    mmseg/datasets/transforms/multimodal_pipelines.py
    #    (skip this if you're happy with the shipped defaults).

Training:

    python tools/train.py configs/floodnet/finetune_sen1floods11_s2.py \\
        --work-dir work_dirs/generalization/sen1floods11_s2/ \\
        --cfg-options \\
            load_from="work_dirs/floodnet/SwinmoeB/655/best_mIoU_epoch_100.pth"
"""

_base_ = ['./finetune_stem_decoder.py']

# ==================== Modal / dataset identifiers ====================
MODAL_NAME = 's2'
MODAL_CHANNELS = 13

ALL_KNOWN_MODALS = {
    # _delete_ forces a full replacement so the base config's
    # sar/rgb/GF entries don't leak through.
    '_delete_': True,
    MODAL_NAME: {
        'channels': MODAL_CHANNELS,
        'pattern': 's2hand',
        'description': 'Sen1Floods11 S2Hand (13-band Sentinel-2 MSI)',
    },
}
TRAINING_MODALS = [MODAL_NAME]
DATASET_NAMES = [MODAL_NAME]

# 512x512 input -> crop to 256x256 for training.
crop_size = (256, 256)

# ==================== Model override ====================
# See finetune_sen1floods11_s1.py for the rationale - the single
# trainable modal here is `s2` with a 13-channel stem conv. The
# s2 mean/std entry in MultiModalNormalize.NORM_CONFIGS is used
# by default; re-run tools/setup_sen1floods11.py to refresh it
# for your local data.
model = dict(
    dataset_names=DATASET_NAMES,
    backbone=dict(
        modal_configs=ALL_KNOWN_MODALS,
        training_modals=TRAINING_MODALS,
    ),
)

# ==================== Dataset / pipelines ====================
dataset_type = 'Sen1Floods11Dataset'
data_root = 'data/Sen1Floods11/'

# Pipeline order matches finetune_sen1floods11_s1.py. See that file
# for the rationale behind each step.
train_pipeline = [
    dict(type='LoadMultiModalImageFromFile', to_float32=True),
    dict(type='LoadSen1Floods11Annotation'),
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
    dict(type='LoadMultiModalImageFromFile', to_float32=True),
    dict(type='LoadSen1Floods11Annotation'),
    dict(type='MultiModalNormalize'),
    dict(type='PackMultiModalSegInputs',
         meta_keys=('img_path', 'ori_filename', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'modal_type', 'actual_channels', 'dataset_name',
                    'img_norm_cfg', 'reduce_zero_label')),
]

# ==================== Dataloaders ====================
# Shares splits/train.txt / splits/val.txt / splits/test.txt with
# finetune_sen1floods11_s1.py so the S1 / S2 results are directly
# comparable (same tiles in each split, just different sensors).
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
        modality=MODAL_NAME,
        ann_file='splits/train.txt',
        data_prefix=dict(
            img_path='S2Hand',
            seg_map_path='LabelHand',
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
        modality=MODAL_NAME,
        ann_file='splits/val.txt',
        data_prefix=dict(
            img_path='S2Hand',
            seg_map_path='LabelHand',
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
        modality=MODAL_NAME,
        ann_file='splits/test.txt',
        data_prefix=dict(
            img_path='S2Hand',
            seg_map_path='LabelHand',
        ),
        pipeline=test_pipeline,
    ),
)
