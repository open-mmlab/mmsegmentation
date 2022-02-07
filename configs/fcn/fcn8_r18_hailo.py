# model settings
_base_ = [
    './fcn_r18_hailo.py',
]
model = dict(
    decode_head=dict(
        in_channels=[128, 256, 512],
        in_index=[1, 2, 3],
    ),
)
