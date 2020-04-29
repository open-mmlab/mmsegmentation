_base_ = './da_r50_encoding_240ep.py'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg), decode_head=dict(norm_cfg=norm_cfg))
