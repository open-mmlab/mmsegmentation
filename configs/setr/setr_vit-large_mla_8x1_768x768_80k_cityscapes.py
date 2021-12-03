_base_ = [
    '../_base_/models/setr_mla.py', '../_base_/datasets/cityscapes_768x768.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained=None,
    backbone=dict(
        drop_rate=0,
        init_cfg=dict(type='Pretrained', checkpoint='mmcls://vit_large_p16')),
    test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(512, 512)))

optimizer = dict(
    lr=0.002,
    weight_decay=0.0,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))
data = dict(samples_per_gpu=1)
