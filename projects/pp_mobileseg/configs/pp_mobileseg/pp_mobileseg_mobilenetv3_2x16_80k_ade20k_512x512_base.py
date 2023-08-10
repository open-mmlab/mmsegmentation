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
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pp_mobileseg/pp_mobileseg_mobilenetv3_3rdparty-base-ed0be681.pth'  # noqa
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size, test_cfg=dict(size_divisor=32))
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(num_classes=150, upsample='intepolate'))
