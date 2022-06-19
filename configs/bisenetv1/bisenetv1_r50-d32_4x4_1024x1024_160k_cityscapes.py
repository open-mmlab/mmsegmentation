_base_ = [
    '../_base_/models/bisenetv1_r18-d32.py',
    '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='BiSeNetV1',
        context_channels=(512, 1024, 2048),
        spatial_channels=(256, 256, 256, 512),
        out_channels=1024,
        backbone_cfg=dict(type='ResNet', depth=50)),
    decode_head=dict(
        type='FCNHead', in_channels=1024, in_index=0, channels=1024),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=512,
            channels=256,
            num_convs=1,
            num_classes=19,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False),
        dict(
            type='FCNHead',
            in_channels=512,
            channels=256,
            num_convs=1,
            num_classes=19,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False),
    ])
param_scheduler = [
    dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=1000),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=1000,
        end=160000,
        by_epoch=False,
    )
]
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
