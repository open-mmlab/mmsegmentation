_base_ = './deeplabv3plus_r101_os16_8x1_110e_cityscapes.py'
model = dict(
    backbone=dict(dilations=(1, 1, 2, 4), strides=(1, 2, 1, 1)),
    decode_head=dict(dilations=(1, 12, 24, 36)))
