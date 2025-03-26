_base_ = [
    '../_base_/models/fast_scnn.py', '../_base_/datasets/hots_v1_640x480.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True, momentum=0.01)
crop_size = (640, 480)
data_preprocessor = dict(size=crop_size)

# Re-config the data sampler.
train_dataloader = dict(batch_size=2, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=2)
test_dataloader = val_dataloader

# Re-config the optimizer.
optimizer = dict(type='SGD', lr=0.12, momentum=0.9, weight_decay=4e-5)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
load_from = "checkpoints/fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853-0cec9937.pth"
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    
    decode_head=dict(num_classes=46),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            channels=32,
            num_convs=1,
            num_classes=46,
            in_index=-2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=64,
            channels=32,
            num_convs=1,
            num_classes=46,
            in_index=-3,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))