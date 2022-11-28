# Copyright (c) OpenMMLab. All rights reserved.
# import torch.nn.functional as F
from typing import List

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import SampleList
from ..losses import accuracy
from ..utils import SelfAttentionBlock, resize
from .decode_head import BaseDecodeHead


class ImageLevelContext(nn.Module):

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
                # weight = F.softmax(preds_iter_cls, dim=0)
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

    def __init__(self, transform_channels, concat_input, shortcut,
                 dropout_ratio, **kwargs):
        super().__init__(**kwargs)

        ilc_cfg = {
            'feat_channels': self.channels,
            'transform_channels': transform_channels,
            'concat_input': concat_input,
            'norm_cfg': self.norm_cfg,
            'act_cfg': self.act_cfg,
            'align_corners': self.align_corners
        }

        slc_cfg = {
            'feat_channels': self.channels,
            'transform_channels': transform_channels,
            'concat_input': concat_input,
            'norm_cfg': self.norm_cfg,
            'act_cfg': self.act_cfg,
        }

        self.ilc_net = ImageLevelContext(**ilc_cfg)
        self.slc_net = SemanticLevelContext(**slc_cfg)

        self.bottleneck = ConvModule(
            self.in_channels,
            self.channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

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

        if shortcut['is_on']:
            self.shortcut = ConvModule(
                shortcut['in_channels'],
                shortcut['feats_channels'],
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

            self.decoder_stage2 = nn.Sequential(
                ConvModule(
                    self.channels + shortcut['feats_channels'],
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

    def _forward_feature(self, inputs):
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
            shortcut_out = self.shortcut(inputs[0])
            feats_sl = resize(
                feats_sl,
                size=shortcut_out.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            feats_sl = torch.cat([feats_sl, shortcut_out], dim=1)
        preds_stage2 = self.decoder_stage2(feats_sl)

        return preds_stage1, preds_stage2

    def forward(self, inputs):
        preds_stage1, preds_stage2 = self._forward_feature(inputs)
        return preds_stage1, preds_stage2

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:

        seg_logits_stage1, seg_logits_stage2 = seg_logits

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        seg_logits_stage1 = resize(
            input=seg_logits_stage1,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_logits_stage2 = resize(
            input=seg_logits_stage2,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits_stage2, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        loss_decode_D = self.loss_decode[0]
        loss_decode_O = self.loss_decode[1]

        loss[loss_decode_D.loss_name] = loss_decode_D(
            seg_logits_stage1,
            seg_label,
            seg_weight,
            ignore_index=self.ignore_index)
        loss[loss_decode_O.loss_name] = loss_decode_O(
            seg_logits_stage2,
            seg_label,
            seg_weight,
            ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logits_stage2, seg_label, ignore_index=self.ignore_index)
        return loss

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        _, seg_logits_stage2 = seg_logits
        seg_logits_stage2 = resize(
            input=seg_logits_stage2,
            size=batch_img_metas[0]['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits_stage2
