_base_ = './ann_r101_torchcv_160ep.py'
norm_cfg = dict(type='NaiveSyncBN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(
        norm_cfg=norm_cfg,
        sampler=dict(type='OHEMSegSampler', thresh=0.7, min_kept=100000)),
    auxiliary_head=dict(norm_cfg=norm_cfg))
