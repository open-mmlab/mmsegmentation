# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .mobilenet_v3_d8_lraspp_4xb4_320k_cityscapes_512x1024 import *

checkpoint = 'open-mmlab://contrib/mobilenet_v3_small'
norm_cfg.update(type=SyncBN, eps=0.001, requires_grad=True)
model.update(
    type=EncoderDecoder,
    backbone=dict(
        type=MobileNetV3,
        init_cfg=dict(type=PretrainedInit, checkpoint=checkpoint),
        arch='small',
        out_indices=(0, 1, 12),
        norm_cfg=norm_cfg),
    decode_head=dict(
        type=LRASPPHead,
        in_channels=(16, 16, 576),
        in_index=(0, 1, 2),
        channels=128,
        input_transform='multiple_select',
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        act_cfg=dict(type=ReLU),
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0)))
