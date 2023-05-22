_base_ = './van-b2_upernet_4xb2-160k_ade20k-512x512.py'
ckpt_path = 'https://download.openmmlab.com/mmsegmentation/v0.5/van_3rdparty/van-b3_3rdparty_20230522-a184e051.pth'  # noqa
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        depths=[3, 5, 27, 3],
        init_cfg=dict(type='Pretrained', checkpoint=ckpt_path),
        drop_path_rate=0.3))
