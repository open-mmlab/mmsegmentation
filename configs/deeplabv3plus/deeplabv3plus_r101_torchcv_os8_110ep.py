_base_ = './deeplabv3plus_r101_torchcv_os16_110ep.py'
model = dict(
    backbone=dict(dilations=(1, 1, 2, 4), strides=(1, 2, 1, 1)),
    decode_head=dict(dilations=(1, 12, 24, 36)))
