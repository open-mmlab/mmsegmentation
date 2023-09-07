_base_ = './catseg_vitl-swin-b_4xb1-warmcoslr2e-4_adamw-80k_coco-stuff164k_384x384.py'  # noqa

model = dict(
    backbone=dict(
        type='CLIPOVCATSeg',
        clip_pretrained='ViT-G',
        custom_clip_weights='~/CLIP-ViT-bigG-14-laion2B-39B-b160k'),
    neck=dict(
        text_guidance_dim=1280,
        appearance_guidance_dim=512,
    ))
