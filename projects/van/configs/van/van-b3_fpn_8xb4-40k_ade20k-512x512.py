_base_ = './van-b2_fpn_8xb4-40k_ade20k-512x512.py'
ckpt_path = 'https://download.openmmlab.com/mmsegmentation/v0.5/van_3rdparty/van-b3_3rdparty_20230522-a184e051.pth'  # noqa
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        embed_dims=[64, 128, 320, 512],
        depths=[3, 5, 27, 3],
        init_cfg=dict(type='Pretrained', checkpoint=ckpt_path),
        drop_path_rate=0.3),
    neck=dict(in_channels=[64, 128, 320, 512]))
train_dataloader = dict(batch_size=4)
