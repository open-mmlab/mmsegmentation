_base_ = [
    '../_base_/models/setr_mla.py', '../_base_/datasets/cityscapes_768x768.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    pretrained=\
    'https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',  # noqa
    backbone=dict(
        drop_rate=0.,
        out_indices=(2, 5, 8, 11),
        embed_dim=768,
        depth=12,
        num_heads=12),
    neck=dict(in_channels=[768, 768, 768, 768]),
    test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(512, 512)))

optimizer = dict(
    lr=0.002,
    weight_decay=0.0,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}),
)

data = dict(samples_per_gpu=1)
