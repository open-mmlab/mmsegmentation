# model settings
_base_ = './fcn_hourglass104.py'
model = dict(
    decode_head=dict(
        in_channels=[256, 256],
        in_index=(0, 1),
        channels=sum([256, 256]),
        input_transform='resize_concat',
        num_convs=2,
        kernel_size=3))
