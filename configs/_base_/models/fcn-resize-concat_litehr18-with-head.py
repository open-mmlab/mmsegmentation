# model settings
_base_ = './fcn-resize-concat_litehr18-without-head.py'
model = dict(
    backbone=dict(extra=dict(with_head=True)),
    decode_head=dict(
        in_channels=[40, 40, 80, 160],
        channels=sum([40, 40, 80, 160]),
    ))
