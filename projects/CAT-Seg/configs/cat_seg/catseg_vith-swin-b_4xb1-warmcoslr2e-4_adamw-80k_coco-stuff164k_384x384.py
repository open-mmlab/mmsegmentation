_base_ = './catseg_vitl-swin-b_4xb1-warmcoslr2e-4_adamw-80k_coco-stuff164k_384x384.py'  # noqa

model = dict(
    backbone=dict(
        type='CLIPOVCATSeg',
        clip_pretrained='ViT-H',
        custom_clip_weights='~/CLIP-ViT-H-14-laion2B-s32B-b79K'),
    neck=dict(
        text_guidance_dim=1024,
        appearance_guidance_dim=512,
    ))
