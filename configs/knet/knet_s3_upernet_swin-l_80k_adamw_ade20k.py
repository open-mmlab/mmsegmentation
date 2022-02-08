_base_ = 'knet_s3_upernet_swin-t_80k_adamw_ade20k.py'

# model settings
model = dict(
    pretrained='./pretrain/swin/swin_large_patch4_window7_224_22k.pth',
    backbone=dict(
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(
        kernel_generate_head=dict(in_channels=[192, 384, 768, 1536])),
    auxiliary_head=dict(in_channels=768))
