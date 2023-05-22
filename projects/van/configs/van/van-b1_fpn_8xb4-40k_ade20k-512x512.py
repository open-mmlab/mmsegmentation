_base_ = './van-b2_fpn_8xb4-40k_ade20k-512x512.py'
ckpt_path = 'https://download.openmmlab.com/mmsegmentation/v0.5/van_3rdparty/van-b1_3rdparty_20230522-3adb117f.pth'  # noqa
model = dict(
    backbone=dict(
        depths=[2, 2, 4, 2],
        init_cfg=dict(type='Pretrained', checkpoint=ckpt_path)))
