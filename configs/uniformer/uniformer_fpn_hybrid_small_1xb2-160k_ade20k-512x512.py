_base_ = './uniformer_fpn_global_small_1xb2-160k_ade20k-512x512.py'
# model settings
model = dict(backbone=dict(type='UniFormer', hybrid=True))
load_from = None
