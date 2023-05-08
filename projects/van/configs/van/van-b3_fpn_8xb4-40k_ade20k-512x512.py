_base_ = './van-b2_fpn_8xb4-40k_ade20k-512x512.py'

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        embed_dims=[64, 128, 320, 512],
        depths=[3, 5, 27, 3],
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/van_b3.pth'),
        drop_path_rate=0.3),
    neck=dict(in_channels=[64, 128, 320, 512]))
train_dataloader = dict(batch_size=4)
