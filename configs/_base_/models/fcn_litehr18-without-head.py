# model settings
_base_ = './fcn-resize-concat_litehr18-without-head.py'
model = dict(
    decode_head=dict(
        in_channels=40,
        in_index=0,
        channels=40,
        input_transform=None,
        kernel_size=3,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1))
