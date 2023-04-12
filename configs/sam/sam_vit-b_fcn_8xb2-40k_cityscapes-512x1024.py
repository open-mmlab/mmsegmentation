_base_ = [
    '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

checkpoint_path = 'https://download.openmmlab.com/mmclassification/v1/vit_sam/vit-base-p16_sam-pre_3rdparty_sa1b-1024px_20230411-2320f9cc.pth'  # noqa
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(1024, 1024),
    test_cfg=dict(size=(1024, 1024)))
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='ViTSAM',
        arch='base',
        img_size=1024,
        patch_size=16,
        out_channels=256,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_path, prefix='backbone')),
    decode_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=0,
        channels=128,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))
