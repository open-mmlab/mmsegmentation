checkpoint_path = 'https://download.openmmlab.com/mmclassification/v1/vit_sam/vit-large-p16_sam-pre_3rdparty_sa1b-1024px_20230411-595feafd.pth'  # noqa
model = dict(
    type='SAM',
    image_encoder_cfg=dict(
        type='ViTSAM',
        arch='large',
        img_size=1024,
        patch_size=16,
        out_channels=256,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_path, prefix='backbone')),
    prompt_encoder_cfg=dict(
        type='PromptEncoder',
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(1024, 1024),
        mask_in_chans=16,
    ),
    mask_decoder_cfg=dict(
        type='MaskDecoder',
        num_multimask_outputs=3,
        transformer=dict(
            type='TwoWayTransformer',
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    ),
    pixel_mean=[123.675, 116.28, 103.53],
    pixel_std=[58.395, 57.12, 57.375])
