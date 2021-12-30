_base_ = ['./segformer_mit-b0_256x256_20k_pot.py']

# model settings
model = dict(
    pretrained='pretrain/mit_b4.pth',
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 8, 27, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))
