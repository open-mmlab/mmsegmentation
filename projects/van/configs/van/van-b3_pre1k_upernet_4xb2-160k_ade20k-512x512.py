_base_ = './van-b2_pre1k_upernet_4xb2-160k_ade20k-512x512.py'

# checkpoint_file = 'https://cloud.tsinghua.edu.cn/d/0100f0cea37d41ba8d08/files/?p=%2Fvan_b3.pth&dl=1'
checkpoint_file = 'pretrained/van_b3.pth'
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        depths=[3, 5, 27, 3],
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        drop_path_rate=0.3))
