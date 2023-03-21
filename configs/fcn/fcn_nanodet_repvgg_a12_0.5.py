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

fpn_cfg =dict(
    name="PAN",
    in_channels=[64, 64, 256],
    out_channels=128,
    start_level=0,
    num_outs=3,
)

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='RepVGG',
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[0.5, 0.5, 0.25, 1.25], 
        override_groups_map=None, deploy=False, fpn_cfg=fpn_cfg),
    decode_head=dict(
        type='ConvHead',
        in_channels=128,
        in_index=0,  # nanodet_repvgg backbone outputs = [batch, 128, 80, 80], [batch, 128, 40, 40], [batch, 128, 20, 20] - this selects [batch, 128, 40, 40], [batch, 128, 20, 20]  for the decode head
        channels=128,
        num_convs=1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    infer_wo_softmax=True)
