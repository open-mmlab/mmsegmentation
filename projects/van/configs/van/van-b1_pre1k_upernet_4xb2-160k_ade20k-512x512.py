_base_ = './van-b2_pre1k_upernet_4xb2-160k_ade20k-512x512.py'

# checkpoint_file = 'https://cloud.tsinghua.edu.cn/d/0100f0cea37d41ba8d08/files/?p=%2Fvan_b1.pth&dl=1'
checkpoint_file = 'pretrained/van_b1.pth'
model = dict(
    backbone=dict(
        depths=[2, 2, 4, 2],
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)
    ))
