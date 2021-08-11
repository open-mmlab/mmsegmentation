norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='pretrain/vit-l_timm.pth', # noqa
    backbone=dict(
        type='VisionTransformer',
        img_size=384,
        embed_dims=1024,
        num_heads=16,
        num_layers=24,
        out_indices=(5, 11, 17, 23),
        final_norm=False,
        with_cls_token=True,
        output_cls_token=True),
    decode_head=dict(
        type='DPTHead',
        in_channels=(1024, 1024, 1024, 1024),
        channels=256,
        embed_dims=1024,
        post_process_channels=[256, 512, 1024, 1024],
        num_classes=150,
        readout_type='project',
        input_transform='multiple_select',
        in_index=(0, 1, 2, 3),
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=None,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable
