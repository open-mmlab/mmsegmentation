"""
Generalization Fine-tuning Config: Single-Modal (RGB only)

Freeze backbone, retrain stem + decoder on a new RGB-only flood event.
Based on finetune_stem_decoder.py but adapted for single-modality data:
  - filter_modality='rgb' to only load RGB images
  - DefaultSampler instead of FixedRatioModalSampler
  - Uses the rgb decode_head from the pretrained separate-decoder model

Usage:
    python tools/train.py configs/floodnet/finetune_single_modal.py \
        --work-dir work_dirs/generalization/LY-train-station/ \
        --cfg-options \
            train_dataloader.dataset.data_root="data/LY-train-station/" \
            val_dataloader.dataset.data_root="data/LY-train-station/" \
            test_dataloader.dataset.data_root="data/LY-train-station/"
"""

_base_ = ['./finetune_stem_decoder.py']

# ==================== Override sampler to DefaultSampler ====================
# _delete_=True forces full replacement instead of recursive merge,
# otherwise FixedRatioModalSampler's modal_ratios etc. leak through
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
        filter_modality='rgb',
    ),
)

val_dataloader = dict(
    dataset=dict(
        filter_modality='rgb',
    ),
)

test_dataloader = dict(
    dataset=dict(
        filter_modality='rgb',
    ),
)
