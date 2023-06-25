# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch import Tensor

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy
from mmseg.models.utils import SelfAttentionBlock, resize
from mmseg.registry import MODELS
from mmseg.utils import SampleList


class ImageLevelContext(nn.Module):
    """ Image-Level Context Module
    Args:
        feats_channels (int): Input channels of query/key feature.
        transform_channels (int): Output channels of key/query transform.
        concat_input (bool): whether to concat input feature.
        align_corners (bool): align_corners argument of F.interpolate.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self,
                 feats_channels,
                 transform_channels,
                 concat_input=False,
                 align_corners=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super().__init__()
        self.align_corners = align_corners
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.correlate_net = SelfAttentionBlock(
            key_in_channels=feats_channels * 2,
            query_in_channels=feats_channels,
            channels=transform_channels,
            out_channels=feats_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=2,
            value_out_num_convs=1,
            key_query_norm=True,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        if concat_input:
            self.bottleneck = ConvModule(
                feats_channels * 2,
                feats_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )

    '''forward'''

    def forward(self, x):
        x_global = self.global_avgpool(x)
        x_global = resize(
            x_global,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        feats_il = self.correlate_net(x, torch.cat([x_global, x], dim=1))
        if hasattr(self, 'bottleneck'):
            feats_il = self.bottleneck(torch.cat([x, feats_il], dim=1))
        return feats_il


class SemanticLevelContext(nn.Module):
    """ Semantic-Level Context Module
    Args:
        feats_channels (int): Input channels of query/key feature.
        transform_channels (int): Output channels of key/query transform.
        concat_input (bool): whether to concat input feature.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self,
                 feats_channels,
                 transform_channels,
                 concat_input=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super().__init__()
        self.correlate_net = SelfAttentionBlock(
            key_in_channels=feats_channels,
            query_in_channels=feats_channels,
            channels=transform_channels,
            out_channels=feats_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=2,
            value_out_num_convs=1,
            key_query_norm=True,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        if concat_input:
            self.bottleneck = ConvModule(
                feats_channels * 2,
                feats_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )

    '''forward'''

    def forward(self, x, preds, feats_il):
        inputs = x
        batch_size, num_channels, h, w = x.size()
        num_classes = preds.size(1)
        feats_sl = torch.zeros(batch_size, h * w, num_channels).type_as(x)
        for batch_idx in range(batch_size):
            # (C, H, W), (num_classes, H, W) --> (H*W, C), (H*W, num_classes)
            feats_iter, preds_iter = x[batch_idx], preds[batch_idx]
            feats_iter, preds_iter = feats_iter.reshape(
                num_channels, -1), preds_iter.reshape(num_classes, -1)
            feats_iter, preds_iter = feats_iter.permute(1,
                                                        0), preds_iter.permute(
                                                            1, 0)
            # (H*W, )
            argmax = preds_iter.argmax(1)
            for clsid in range(num_classes):
                mask = (argmax == clsid)
                if mask.sum() == 0:
                    continue
                feats_iter_cls = feats_iter[mask]
                preds_iter_cls = preds_iter[:, clsid][mask]
                weight = torch.softmax(preds_iter_cls, dim=0)
                feats_iter_cls = feats_iter_cls * weight.unsqueeze(-1)
                feats_iter_cls = feats_iter_cls.sum(0)
                feats_sl[batch_idx][mask] = feats_iter_cls
        feats_sl = feats_sl.reshape(batch_size, h, w, num_channels)
        feats_sl = feats_sl.permute(0, 3, 1, 2).contiguous()
        feats_sl = self.correlate_net(inputs, feats_sl)
        if hasattr(self, 'bottleneck'):
            feats_sl = self.bottleneck(torch.cat([feats_il, feats_sl], dim=1))
        return feats_sl


@MODELS.register_module()
class ISNetHead(BaseDecodeHead):
    """ISNet: Integrate Image-Level and Semantic-Level
    Context for Semantic Segmentation

    This head is the implementation of `ISNet`
    <https://arxiv.org/pdf/2108.12382.pdf>`_.

    Args:
        transform_channels (int): Output channels of key/query transform.
        concat_input (bool): whether to concat input feature.
        with_shortcut (bool): whether to use shortcut connection.
        shortcut_in_channels (int): Input channels of shortcut.
        shortcut_feat_channels (int): Output channels of shortcut.
        dropout_ratio (float): Ratio of dropout.
    """

    def __init__(self, transform_channels, concat_input, with_shortcut,
                 shortcut_in_channels, shortcut_feat_channels, dropout_ratio,
                 **kwargs):
        super().__init__(**kwargs)

        self.in_channels = self.in_channels[-1]

        self.bottleneck = ConvModule(
            self.in_channels,
            self.channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.ilc_net = ImageLevelContext(
            feats_channels=self.channels,
            transform_channels=transform_channels,
            concat_input=concat_input,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.slc_net = SemanticLevelContext(
            feats_channels=self.channels,
            transform_channels=transform_channels,
            concat_input=concat_input,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.decoder_stage1 = nn.Sequential(
            ConvModule(
                self.channels,
                self.channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            nn.Dropout2d(dropout_ratio),
            nn.Conv2d(
                self.channels,
                self.num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True),
        )

        if with_shortcut:
            self.shortcut = ConvModule(
                shortcut_in_channels,
                shortcut_feat_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            self.decoder_stage2 = nn.Sequential(
                ConvModule(
                    self.channels + shortcut_feat_channels,
                    self.channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                nn.Dropout2d(dropout_ratio),
                nn.Conv2d(
                    self.channels,
                    self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True),
            )
        else:
            self.decoder_stage2 = nn.Sequential(
                nn.Dropout2d(dropout_ratio),
                nn.Conv2d(
                    self.channels,
                    self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True),
            )

        self.conv_seg = None
        self.dropout = None

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        feats = self.bottleneck(x[-1])

        feats_il = self.ilc_net(feats)

        preds_stage1 = self.decoder_stage1(feats)
        preds_stage1 = resize(
            preds_stage1,
            size=feats.size()[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        feats_sl = self.slc_net(feats, preds_stage1, feats_il)

        if hasattr(self, 'shortcut'):
            shortcut_out = self.shortcut(x[0])
            feats_sl = resize(
                feats_sl,
                size=shortcut_out.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            feats_sl = torch.cat([feats_sl, shortcut_out], dim=1)
        preds_stage2 = self.decoder_stage2(feats_sl)

        return preds_stage1, preds_stage2

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits[-1], seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        for seg_logit, loss_decode in zip(seg_logits, self.loss_decode):
            seg_logit = resize(
                input=seg_logit,
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            loss[loss_decode.name] = loss_decode(
                seg_logit,
                seg_label,
                seg_weight,
                ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logits[-1], seg_label, ignore_index=self.ignore_index)
        return loss

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        _, seg_logits_stage2 = seg_logits
        return super().predict_by_feat(seg_logits_stage2, batch_img_metas)
