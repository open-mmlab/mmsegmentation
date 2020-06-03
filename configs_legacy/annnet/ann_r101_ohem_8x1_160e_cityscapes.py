_base_ = './ann_r101_8x1_160e_cityscapes.py'
model = dict(
    decode_head=dict(
        sampler=dict(type='OHEMSegSampler', thresh=0.7, min_kept=100000)))
