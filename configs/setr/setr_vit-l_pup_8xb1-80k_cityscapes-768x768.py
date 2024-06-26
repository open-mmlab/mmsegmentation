_base_ = [
    '../_base_/models/setr_pup.py', '../_base_/datasets/cityscapes_768x768.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (768, 768)
data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (768, 768)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        drop_rate=0.,
        init_cfg=dict(
            type='Pretrained', checkpoint="checkpoints/vit-large-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-b20ba619.pth")),
    auxiliary_head=[
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=0,
            num_classes=19,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            num_convs=2,
            up_scale=4,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=1,
            num_classes=19,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            num_convs=2,
            up_scale=4,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=2,
            num_classes=19,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            num_convs=2,
            up_scale=4,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    ],
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(512, 512)))

optimizer = dict(weight_decay=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))

train_dataloader = dict(batch_size=1)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
