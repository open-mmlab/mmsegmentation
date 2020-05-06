_base_ = './deeplabv3plus_xcp65_segmentron_os16_110ep.py'
model = dict(backbone=dict(output_stride=32))
