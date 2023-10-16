_base_ = './uniformer_upernet_global_base_1xb2-160k_ade20k-512x512.py'
model = dict(backbone=dict(hybrid=True))
load_from = None
