_base_ = './van-b2_pre1k_upernet_4xb2-160k_ade20k-512x512.py'

# checkpoint_file = 'https://cloud.tsinghua.edu.cn/d/0100f0cea37d41ba8d08/files/?p=%2Fvan_b0.pth&dl=1'
checkpoint_file = 'pretrained/van_b0.pth'
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)
    ),
    decode_head=dict(in_channels=[32, 64, 160, 256],
                     num_classes=150
    ),
    auxiliary_head=dict(in_channels=160, num_classes=150)
)
