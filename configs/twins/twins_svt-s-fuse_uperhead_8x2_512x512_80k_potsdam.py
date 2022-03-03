_base_ = [
    '../_base_/models/twins_pcpvt-s_upernet.py',
    '../_base_/datasets/Potsdam_ndsm.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
model = dict(
    backbone=dict(
        type='EDFT',
        backbone="Twins_svt",
        in_channels=4,
        weight=0.4,
        overlap=True,
        attention_type='dsa-add',
        same_branch=False,
        init_cfg=dict(
            type='Pretrained', checkpoint='pretrain/alt_gvt_small.pth'),
        embed_dims=[64, 128, 256, 512],
        num_heads=[2, 4, 8, 16],
        mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 10, 4],
        windiow_sizes=[7, 7, 7, 7],
        norm_after_stage=True),
    decode_head=dict(in_channels=[64, 128, 256, 512], num_classes=6),
    auxiliary_head=dict(in_channels=256, num_classes=6),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(custom_keys={
        'pos_block': dict(decay_mult=0.),
        'norm': dict(decay_mult=0.)
    }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=2, workers_per_gpu=2)
evaluation = dict(metric=['mIoU', 'mFscore'], save_best='mIoU')
