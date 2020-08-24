_base_ = [
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
)

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True, momentum=0.01)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='MobileNetv3',
        arch='big',
        norm_cfg=norm_cfg,
        out_indices=(5, 12),  # 1/8 res and 1/16 res.
    ),
    decode_head=dict(
        type='LR_ASPPHead',
        in_channels=(40, 160),
        input_transform='multiple_select',
        channels=19,
        num_classes=19,
        in_index=(0, 1),
        # all the outputs of MobileNetV3 backbone are needed.
        norm_cfg=norm_cfg,
        act_cfg=dict(type='HSwish'),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.)),
)

# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')

# redefine optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.00001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
total_iters = 160000
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU')
