# Copyright (c) OpenMMLab. All rights reserved.
"""CM-DGSeg B2 configuration for Cityscapes main results.

Run command:
    python tools/train.py configs/cm-dgseg/cm_dgseg_b2_cityscapes.py
    python tools/test.py configs/cm-dgseg/cm_dgseg_b2_cityscapes.py <checkpoint> --eval mIoU
"""

_base_ = [
    './cm_dgseg_b0_cityscapes.py',
]

# Switch MiT backbone to B2 and tighten optimization for the final run.
model = dict(
    backbone_rgb=dict(
        init_cfg=dict(type='Pretrained', checkpoint='{{PRETRAIN_B2}}'),
        embed_dims=64,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        drop_path_rate=0.2),
    backbone_disp=dict(
        embed_dims=32,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        drop_path_rate=0.1),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        disp_in_channels=[32, 64, 160, 256],
        channels=512,  # set to 256 if GPU memory is limited.
        dropout_ratio=0.0,
        with_boundary=True,
        boundary_loss_weight=0.2,
        geometry_reg_weight=0.05))

# Training tweaks for the sprint setting.
# train_dataloader = dict(batch_size=4, num_workers=8)
# optim_wrapper = dict(
#     optimizer=dict(lr=5e-5),
#     paramwise_cfg=dict(
#         custom_keys=dict(
#             head=dict(lr_mult=12.),
#             backbone_disp=dict(lr_mult=0.5))))

# param_scheduler = [
#     dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
#     dict(
#         type='PolyLR',
#         eta_min=0.0,
#         power=1.0,
#         begin=1500,
#         end=160000,
#         by_epoch=False)
# ]

# train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=8000)
# default_hooks = dict(checkpoint=dict(interval=8000, max_keep_ckpts=5))
