_base_ = './stdc-813_512x1024_80k_cityscapes.py'
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(
        type='STDCContextPathNet',
        stdc_cfg=dict(
            type='STDCNet',
            stdc_type='STDCNet1446',
            in_channels=3,
            channels=(32, 64, 256, 512, 1024),
            bottleneck_type='cat',
            stdc_num_convs=4,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            with_final_conv=False),
        last_in_channels=(512, 1024),
        out_channels=128,
        ffm_cfg=dict(in_channels=384, out_channels=256, scale_factor=4)))
