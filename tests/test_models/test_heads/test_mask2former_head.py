# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine import Config
from mmengine.structures import PixelData

from mmseg.models.decode_heads import Mask2FormerHead
from mmseg.structures import SegDataSample
from mmseg.utils import SampleList
from .utils import to_cuda


def test_mask2former_head():
    num_classes = 19
    cfg = dict(
        in_channels=[96, 192, 384, 768],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(  # DeformableDetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None),
            positional_encoding=dict(  # SinePositionalEncoding
                num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(  # SinePositionalEncoding
            num_feats=128, normalize=True),
        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True)),
            init_cfg=None),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(
                        type='mmdet.CrossEntropyLossCost',
                        weight=5.0,
                        use_sigmoid=True),
                    dict(
                        type='mmdet.DiceCost',
                        weight=5.0,
                        pred_act=True,
                        eps=1.0)
                ]),
            sampler=dict(type='mmdet.MaskPseudoSampler')))
    cfg = Config(cfg)
    head = Mask2FormerHead(**cfg)

    inputs = [
        torch.rand((2, 96, 8, 8)),
        torch.rand((2, 192, 4, 4)),
        torch.rand((2, 384, 2, 2)),
        torch.rand((2, 768, 1, 1))
    ]

    data_samples: SampleList = []
    for i in range(2):
        data_sample = SegDataSample()
        img_meta = {}
        img_meta['img_shape'] = (32, 32)
        img_meta['ori_shape'] = (32, 32)
        data_sample.gt_sem_seg = PixelData(
            data=torch.randint(0, num_classes, (1, 32, 32)))
        data_sample.set_metainfo(img_meta)
        data_samples.append(data_sample)

    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
        for data_sample in data_samples:
            data_sample.gt_sem_seg.data = data_sample.gt_sem_seg.data.cuda()

    loss_dict = head.loss(inputs, data_samples, None)
    assert isinstance(loss_dict, dict)

    batch_img_metas = []
    for data_sample in data_samples:
        batch_img_metas.append(data_sample.metainfo)

    seg_logits = head.predict(inputs, batch_img_metas, None)
    assert seg_logits.shape == torch.Size((2, num_classes, 32, 32))
