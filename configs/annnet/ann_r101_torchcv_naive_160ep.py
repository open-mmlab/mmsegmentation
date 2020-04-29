_base_ = './ann_r101_torchcv_160ep.py'
norm_cfg = dict(type='NaiveSyncBN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(norm_cfg=norm_cfg),
    auxiliary_head=dict(norm_cfg=norm_cfg))
