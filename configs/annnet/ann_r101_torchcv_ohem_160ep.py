_base_ = './ann_r101_torchcv_160ep.py'
model = dict(
    decode_head=dict(
        sampler=dict(type='OHEMSegSampler', thresh=0.7, min_kept=100000)))
