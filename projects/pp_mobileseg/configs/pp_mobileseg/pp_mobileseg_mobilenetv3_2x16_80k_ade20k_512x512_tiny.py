_base_ = [
    '../_base_/models/pp_mobile.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
# the custom import path is determined by your workspace path (i.e., where you run the command from) # noqa
custom_imports = dict(
    imports=[
        'projects.pp_mobileseg.backbones', 'projects.pp_mobileseg.decode_head'
    ],
    allow_failed_imports=False)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pp_mobileseg/pp_mobileseg_mobilenetv3_3rdparty-tiny-e4b35e96.pth'  # noqa
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size, test_cfg=dict(size_divisor=32))
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        type='StrideFormer',
        mobileV3_cfg=[
            # k t c, s
            [[3, 16, 16, True, 'ReLU', 1], [3, 64, 32, False, 'ReLU', 2],
             [3, 48, 24, False, 'ReLU', 1]],  # cfg1
            [[5, 96, 32, True, 'HSwish', 2], [5, 96, 32, True, 'HSwish',
                                              1]],  # cfg2
            [[5, 160, 64, True, 'HSwish', 2], [5, 160, 64, True, 'HSwish',
                                               1]],  # cfg3
            [[3, 384, 128, True, 'HSwish', 2],
             [3, 384, 128, True, 'HSwish', 1]],  # cfg4
        ],
        channels=[16, 24, 32, 64, 128],
        depths=[2, 2],
        embed_dims=[64, 128],
        num_heads=4,
        inj_type='AAM',
        out_feat_chs=[32, 64, 128],
        act_cfg=dict(type='ReLU6'),
    ),
    decode_head=dict(
        num_classes=150,
        in_channels=256,
        use_dw=True,
        act_cfg=dict(type='ReLU'),
        upsample='intepolate'),
)
