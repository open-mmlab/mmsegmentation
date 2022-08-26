_base_ = ['./segformer_mit-b0_8xb2-160k_ade20k-512x512.py']

# model settings
model = dict(
    pretrained='pretrain/mit_b2.pth',
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 4, 6, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))
