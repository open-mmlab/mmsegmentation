_base_ = './stdc-813_512x1024_80k_cityscapes.py'
model = dict(backbone=dict(stdc_cfg=dict(stdc_type='STDCNet1446')))
