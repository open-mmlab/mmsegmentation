# model settings
_base_ = [
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
]
# optimizer
optimizer = dict(type='Adam', lr=1e-3, weight_decay=1e-5)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='CosineAnnealing', warmup='linear',
                 min_lr=1e-6, by_epoch=True, warmup_iters=5, warmup_ratio=0.2)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=40)
checkpoint_config = dict(by_epoch=True, interval=5)
evaluation = dict(interval=5, metric='mIoU', pre_eval=True)

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='torchvision://resnet18',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='FCNGenHead',
        in_channels=[256, 512],
        input_transform='multiple_select',
        in_index=[2, 3],
        channels=512,
        num_convs=0,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
