_base_ = './fpn_poolformer_s12_8x4_ade20k_40k.py'

# model settings
model = dict(
    backbone=dict(arch='m48'), neck=dict(in_channels=[96, 192, 384, 768]))
