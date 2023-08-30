# model settings
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='VPD',
        diffusion_cfg=dict(
            config=
            'vpd/stable-diffusion/configs/stable-diffusion/v1-inference.yaml',
            checkpoint='vpd/v1-5-pruned-emaonly.ckpt',
        ),
        class_embed_path='vpd/class_embeddings.pth'),
)
