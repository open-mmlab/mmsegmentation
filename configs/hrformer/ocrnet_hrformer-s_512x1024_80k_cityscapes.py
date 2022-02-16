_base_ = [
    '../_base_/models/ocrnet_hrformer-s.py',
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True, momentum=0.1)
model = dict(
    pretrained='pretrain/hrt_small.pth',
    backbone=dict(
        type='HRFormer',
        norm_cfg=norm_cfg,
        norm_eval=False,
        drop_path_rate=0.2,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(2, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='HRFORMER',
                window_sizes=(7, 7),
                num_heads=(1, 2),
                mlp_ratios=(4, 4),
                num_blocks=(2, 2),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='HRFORMER',
                window_sizes=(7, 7, 7),
                num_heads=(1, 2, 4),
                mlp_ratios=(4, 4, 4),
                num_blocks=(2, 2, 2),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=2,
                num_branches=4,
                block='HRFORMER',
                window_sizes=(7, 7, 7, 7),
                num_heads=(1, 2, 4, 8),
                mlp_ratios=(4, 4, 4, 4),
                num_blocks=(2, 2, 2, 2),
                num_channels=(32, 64, 128, 256)))),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[32, 64, 128, 256],
            channels=sum([32, 64, 128, 256]),
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            kernel_size=1,
            num_convs=1,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=150,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[32, 64, 128, 256],
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            channels=512,
            ocr_channels=256,
            dropout_ratio=-1,
            num_classes=150,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            sampler=dict(type='OHEMPixelSampler', thresh=0.9, min_kept=100000)),
    ],
)
# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
