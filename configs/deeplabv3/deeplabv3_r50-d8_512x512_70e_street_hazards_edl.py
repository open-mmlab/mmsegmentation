_base_ = [
    '../_base_/models/deeplabv3_r50-d8_edl.py',
    '../_base_/datasets/street_hazards_512x512.py', '../_base_/epoch_runtime.py',
    '../_base_/schedules/schedule_70e.py'
]
model = dict(
    decode_head=dict(align_corners=True, num_classes=12,
                     loss_decode=dict(type='EDLLoss', num_classes=12)),
    # auxiliary_head=dict(align_corners=True),
    test_cfg=dict(mode='whole')
)
