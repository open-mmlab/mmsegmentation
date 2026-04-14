"""
Sen1Floods11 fine-tune: S2Hand (13-band Sentinel-2 MSI).

Sibling of ``finetune_sen1floods11_s1.py`` - same freeze / LR / schedule
strategy, but swaps the trainable modal to ``s2`` (13ch) and points the
dataset at the ``S2Hand`` subdirectory. LabelHand files are shared
between S1 and S2; the signed -1 nodata is remapped to ignore_index.

Usage::

    python tools/train.py configs/floodnet/finetune_sen1floods11_s2.py \
        --work-dir work_dirs/generalization/sen1floods11_s2/ \
        --cfg-options \
            load_from="work_dirs/floodnet/SwinmoeB/655/best_mIoU_epoch_100.pth" \
            train_dataloader.dataset.data_root="data/Sen1Floods11/" \
            val_dataloader.dataset.data_root="data/Sen1Floods11/" \
            test_dataloader.dataset.data_root="data/Sen1Floods11/"
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
        'description': 'Sen1Floods11 S2Hand (13-band Sentinel-2)',
    },
}
TRAINING_MODALS = [MODAL_NAME]
DATASET_NAMES = [MODAL_NAME]

crop_size = (256, 256)

# ==================== Model override ====================
# See finetune_sen1floods11_s1.py for the rationale - the single
# trainable modal here is `s2` with a 13-channel stem conv.
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

train_pipeline = [
    dict(type='LoadMultiModalImageFromFile', to_float32=True),
    dict(type='LoadSen1Floods11Annotation'),
    dict(type='RandomResize',
         scale=(2048, 512),
         ratio_range=(0.5, 2.0),
         keep_ratio=True),
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
    dict(type='MultiModalNormalize'),
    dict(type='LoadSen1Floods11Annotation'),
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
        modality=MODAL_NAME,
        data_prefix=dict(
            img_path='S2Hand',
            seg_map_path='LabelHand',
        ),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        modality=MODAL_NAME,
        data_prefix=dict(
            img_path='S2Hand',
            seg_map_path='LabelHand',
        ),
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        modality=MODAL_NAME,
        data_prefix=dict(
            img_path='S2Hand',
            seg_map_path='LabelHand',
        ),
        pipeline=test_pipeline,
    ),
)
