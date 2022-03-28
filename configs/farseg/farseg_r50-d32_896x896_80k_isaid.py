_base_ = [
    '../_base_/datasets/isaid.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    neck=dict(
        type='FARNeck',
        neck_cfg=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=4),
        scene_relation_in_channels=2048,
        scene_relation_channel_list=(256, 256, 256, 256),
        scene_relation_out_channels=256,
        scale_aware_proj=True,
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        type='FARHead',
        in_channels=256,
        out_channels=128,
        in_feat_output_strides=(4, 8, 16, 32),
        out_feat_output_stride=4,
        norm_cfg=norm_cfg,
        num_classes=16,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

data = dict(samples_per_gpu=2, workers_per_gpu=2)

# optimizer
optimizer = dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)
