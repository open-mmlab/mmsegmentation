_base_ = './deeplabv3plus_xcp65_os16_8x1_110e_cityscapes.py'
model = dict(backbone=dict(output_stride=32))
