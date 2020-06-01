_base_ = [
    '../_base_/models/fcn_r50.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_60ki.py'
]
model = dict(
    decode_head=dict(
        sampler=dict(type='OHEMSegSampler', thresh=0.7, min_kept=100000)))
