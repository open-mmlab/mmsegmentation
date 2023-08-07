_base_ = [
    '../_base_/models/pp_mobile.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
checkpoint = './models/pp_mobile_base.pth'
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size, test_cfg=dict(size_divisor=32))
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint), ),
    decode_head=dict(num_classes=150, upsample='intepolate'),
)
