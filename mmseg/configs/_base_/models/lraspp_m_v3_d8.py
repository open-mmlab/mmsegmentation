# Copyright (c) OpenMMLab. All rights reserved.
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import SyncBatchNorm as SyncBN

from mmseg.models.backbones import MobileNetV3
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.models.decode_heads import LRASPPHead
from mmseg.models.losses import CrossEntropyLoss
from mmseg.models.segmentors import EncoderDecoder

# model settings
norm_cfg = dict(type=SyncBN, eps=0.001, requires_grad=True)
data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type=MobileNetV3,
        arch='large',
        out_indices=(1, 3, 16),
        norm_cfg=norm_cfg),
    decode_head=dict(
        type=LRASPPHead,
        in_channels=(16, 24, 960),
        in_index=(0, 1, 2),
        channels=128,
        input_transform='multiple_select',
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        act_cfg=dict(type=ReLU),
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
