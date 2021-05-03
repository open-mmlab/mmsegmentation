_base_ = [
    '../_base_/models/setr_mla.py', '../_base_/datasets/cityscapes_768x768.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained='https://dl.fbaipublicfiles.com/deit/\
deit_base_distilled_patch16_384-d0272ac0.pth',
    backbone=dict(drop_rate=0.))

optimizer = dict(
    lr=0.002,
    weight_decay=0.0,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}),
    test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(512, 512)))

find_unused_parameters = True
data = dict(samples_per_gpu=1)
