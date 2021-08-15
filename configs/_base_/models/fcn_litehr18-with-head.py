# model settings
_base_ = './fcn_litehr18-without-head.py'
model = dict(backbone=dict(extra=dict(with_head=True)))
