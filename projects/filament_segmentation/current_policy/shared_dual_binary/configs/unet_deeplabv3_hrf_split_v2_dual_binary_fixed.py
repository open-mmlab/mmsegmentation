# One shared DeepLabV3/UNet model with two binary output heads.
#
# Place this file at:
#   configs/unet/unet_deeplabv3_hrf_split_v2_dual_binary.py
#
# It inherits the current confirmed config so that existing paths,
# optimizer, scheduler, HRF-pretrained UNet backbone and DeepLabV3 head
# stay unchanged as much as possible.
#
# IMPORTANT:
# The model must receive the ORIGINAL 3-class mask:
#   0 background, 1 confirmed, 2 candidate.
# Therefore the old binary-label transform must NOT be used here.

_base_ = [
    '../../independent_binary/configs/unet_deeplabv3_hrf_split_v2_background_vs_confirmed.py'
]

custom_imports = dict(
    imports=[
        'projects.filament_segmentation.current_policy.shared_dual_binary.models.dual_binary_encoder_decoder_fixed_metric',
    ],
    allow_failed_imports=False,
)

model = dict(
    type='DualBinaryEncoderDecoder',
    confirmed_label=1,
    candidate_label=2,
    ignore_index=255,
    confirmed_loss_weight=1.0,
    candidate_loss_weight=1.0,
    decode_head=dict(
        num_classes=2,
    ),
)

# Raw 3-class masks are loaded directly.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

# Existing dataset paths/classes are inherited.
train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
    ),
)

# Training/validation only.
# The inherited base config has no complete TEST loop, so keep all three
# TEST-related settings None. This avoids MMEngine's consistency error:
# test_dataloader, test_cfg and test_evaluator must be all None or all set.
test_dataloader = None
test_cfg = None
test_evaluator = None

val_evaluator = dict(
    _delete_=True,
    type='DualBinarySegMetric',
    confirmed_label=1,
    candidate_label=2,
    ignore_index=255,
)

work_dir = 'projects/filament_segmentation/current_policy/shared_dual_binary/work_dirs/unet_deeplabv3_hrf_split_v2_dual_binary'
