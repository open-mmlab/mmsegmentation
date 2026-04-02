"""
Generalization Fine-tuning Config: Freeze Backbone, Retrain Stem + Decoder

Loads pretrained Full Model checkpoint, freezes all 4 Swin Transformer stages
(attention, MoE, patch merging) via backbone.frozen_stages=3, and only retrains:
  1. ModalSpecificPatchEmbed (backbone.patch_embed) - adapts to new sensor inputs
  2. UPerHead decode_heads - adapts segmentation output to new domain
  3. FCNHead auxiliary_heads - auxiliary segmentation head

Frozen stages use requires_grad=False + eval() mode for proper BN/Dropout behavior.

Usage:
    python tools/train.py configs/floodnet/finetune_stem_decoder.py \
        --work-dir work_dirs/generalization/event_name/ \
        --cfg-options \
            load_from="work_dirs/floodnet/SwinmoeB/655/best_mIoU_epoch_100.pth" \
            train_dataloader.dataset.data_root="../floodnet/data/new_event/" \
            val_dataloader.dataset.data_root="../floodnet/data/new_event/" \
            test_dataloader.dataset.data_root="../floodnet/data/new_event/"
"""

_base_ = ['./multimodal_floodnet_sar_boost_swinbase_moe_config.py']

# ==================== Load pretrained checkpoint ====================
# Override this via --cfg-options load_from="path/to/checkpoint.pth"
load_from = 'work_dirs/floodnet/SwinmoeB/655/best_mIoU_epoch_100.pth'

# ==================== Freeze all 4 backbone stages ====================
# frozen_stages=3 freezes stages 0-3 (all Swin blocks, MoE, patch merging, norms)
# freeze_patch_embed=False keeps ModalSpecificPatchEmbed trainable
model = dict(
    backbone=dict(
        frozen_stages=3,         # Freeze all 4 stages (0,1,2,3)
        freeze_patch_embed=False  # Keep stem trainable
    ),
)

# ==================== Optimizer for fine-tuning ====================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # Higher base LR for fine-tuning fewer params
        betas=(0.9, 0.999),
        weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            # Stem: moderate LR boost
            'backbone.patch_embed': dict(lr_mult=2.0),
            # Decode heads: high LR for fast adaptation
            'decode_heads': dict(lr_mult=10.0),
            'auxiliary_heads': dict(lr_mult=10.0),
        }
    )
)

# ==================== Shorter schedule for fine-tuning ====================
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10),  # Short warmup
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=2,
        end=30,
        by_epoch=True),
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,  # Much shorter than full training (100 epochs)
    val_interval=10)

# ==================== Checkpoint hook ====================
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=10,
        max_keep_ckpts=1,
        save_best='mIoU'),
)
