_base_ = [
    '../_base_/models/vpd_sd.py', '../_base_/datasets/nyu.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_25k.py'
]

crop_size = (480, 480)

model = dict(
    type='DepthEstimator',
    data_preprocessor=dict(size=crop_size),
    backbone=dict(
        class_embed_path='https://download.openmmlab.com/mmsegmentation/'
        'v0.5/vpd/nyu_class_embeddings.pth',
        class_embed_select=True,
        pad_shape=512,
        unet_cfg=dict(use_attn=False),
    ),
    decode_head=dict(
        type='VPDDepthHead',
        in_channels=[320, 640, 1280, 1280],
        max_depth=10,
        fmap_border=(1, 1),
    ),
    test_cfg=dict(mode='slide_flip', crop_size=crop_size, stride=(160, 160)))

default_hooks = dict(
    checkpoint=dict(save_best='rmse', rule='less', max_keep_ckpts=1))

# custom optimizer
optim_wrapper = dict(
    constructor='ForceDefaultOptimWrapperConstructor',
    paramwise_cfg=dict(
        bias_decay_mult=0,
        force_default_settings=True,
        custom_keys={
            'backbone.encoder_vq': dict(lr_mult=0),
            'backbone.unet': dict(lr_mult=0.01),
        }))
