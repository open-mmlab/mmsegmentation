_base_ = './dpt_vit-b16_512x512_160k_ade20k.py'

model = dict(
    pretrained='pretrain/vit-l16_p16_384.pth', # noqa
    backbone=dict(
        img_size=384,
        embed_dims=1024,
        num_heads=16,
        num_layers=24,
        out_indices=(5, 11, 17, 23)),
    decode_head=dict(
        in_channels=(1024, 1024, 1024, 1024),
        embed_dims=1024,
        post_process_channels=[256, 512, 1024, 1024]))  # yapf: disable
