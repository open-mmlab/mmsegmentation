_base_ = './dpt_vit-b16_512x512_160k_ade20k.py'

model = dict(
    type='EncoderDecoder',
    pretrained='pretrain/vit-l16_p16_384.pth', # noqa
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
        post_process_channels=[256, 512, 1024, 1024]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable
