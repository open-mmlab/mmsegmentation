_base_ = './stdc1_512x1024_80k_cityscapes.py'
model = dict(backbone=dict(backbone_cfg=dict(stdc_type='STDCNet2')))
