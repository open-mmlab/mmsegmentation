_base_ = './stdc1_4xb12-80k_cityscapes-512x1024.py'
model = dict(backbone=dict(backbone_cfg=dict(stdc_type='STDCNet2')))
