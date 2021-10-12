_base_ = [
    '../_base_/models/segmenter_vit.py',
    '../_base_/datasets/cityscapes_769x769.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py',
]

backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='pretrain/vit_base_p16_384.pth',
    backbone=dict(
        type='VisionTransformer',
        img_size=(768, 768),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        final_norm=True,
        norm_cfg=backbone_norm_cfg,
        with_cls_token=True,
        interpolate_mode='bicubic',
    ),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=768,
        channels=768,
        num_classes=19,
        num_layers=2,
        num_heads=12,
        embed_dims=768,
        in_index=-1,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    auxiliary_head=[],
    test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(768, 768)),
)

optimizer = dict(lr=0.01, weight_decay=0.0)

# num_gpus: 8 -> batch_size: 8
data = dict(samples_per_gpu=1, )

# TODO: handle img_norm_cfg
# img_norm_cfg = dict(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
