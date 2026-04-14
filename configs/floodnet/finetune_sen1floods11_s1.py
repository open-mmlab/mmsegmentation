"""
Sen1Floods11 fine-tune: S1Hand (2-band VV/VH SAR).

Mirrors ``configs/floodnet/finetune_single_modal.py`` but switches to
the :class:`Sen1Floods11Dataset` + :class:`LoadSen1Floods11Annotation`
loader so the 2-band SAR TIFFs and signed-int LabelHand files (where
-1 = nodata) are handled correctly.

The pretrained Swin-Base + MoE checkpoint has modal-specific patch
embeds / decode heads for ``sar / rgb / GF``. Here we redefine the
single trainable modal as ``s1`` (2ch) and the single decode-head key
as ``s1``; the mismatched pretrained keys are ignored on load
(``strict=False``), so the new patch-embed and decode head train from
scratch while the frozen Swin stages reuse their pretrained weights.

Usage::

    python tools/train.py configs/floodnet/finetune_sen1floods11_s1.py \
        --work-dir work_dirs/generalization/sen1floods11_s1/ \
        --cfg-options \
            load_from="work_dirs/floodnet/SwinmoeB/655/best_mIoU_epoch_100.pth" \
            train_dataloader.dataset.data_root="data/Sen1Floods11/" \
            val_dataloader.dataset.data_root="data/Sen1Floods11/" \
            test_dataloader.dataset.data_root="data/Sen1Floods11/"
"""

_base_ = ['./finetune_stem_decoder.py']

# ==================== Modal / dataset identifiers ====================
MODAL_NAME = 's1'
MODAL_CHANNELS = 2

ALL_KNOWN_MODALS = {
    # _delete_ forces a full replacement so the base config's
    # sar/rgb/GF entries don't leak through.
    '_delete_': True,
    MODAL_NAME: {
        'channels': MODAL_CHANNELS,
        'pattern': 's1hand',
        'description': 'Sen1Floods11 S1Hand (VV/VH dB)',
    },
}
TRAINING_MODALS = [MODAL_NAME]
DATASET_NAMES = [MODAL_NAME]

crop_size = (256, 256)

# ==================== Model override ====================
# - Restrict the modal-specific patch embed to only `s1` (2ch conv).
# - Restrict the per-dataset decode head ModuleDict to just `s1`.
# Pretrained weights for sar/rgb/GF patch embeds, decode heads, and
# the [3, num_experts] MoE modal_bias are all shape-mismatched vs the
# new single-modal model, so mmengine's load_checkpoint
# (strict=False) warns and leaves those tensors at their fresh init.
# The frozen Swin body still reuses the pretrained attention / MoE
# expert weights, which is what supplies the cross-domain priors.
model = dict(
    dataset_names=DATASET_NAMES,
    backbone=dict(
        modal_configs=ALL_KNOWN_MODALS,
        training_modals=TRAINING_MODALS,
        # frozen_stages / freeze_patch_embed inherit from base
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
# _delete_=True on sampler forces full replacement so the base
# FixedRatioModalSampler config is discarded - Sen1Floods11 is single
# modal, so we just use the standard shuffling sampler.
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
            img_path='S1Hand',
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
            img_path='S1Hand',
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
            img_path='S1Hand',
            seg_map_path='LabelHand',
        ),
        pipeline=test_pipeline,
    ),
)
