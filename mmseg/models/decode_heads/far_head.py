# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.losses import cross_entropy
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class FARHead(BaseDecodeHead):
    """FarSeg Head.

    This head is the implementation of Light-weight Decoder in
    `Foreground-Aware Relation Network for Geospatial Object Segmentation
    in High Spatial Resolution Remote Sensing Imagery
    <https://arxiv.org/abs/2011.09766>`_.

    Args:
        in_channels (int): The number of input channels.
            Default: 256.
        out_channels (int): The number of output channels.
            Default: 128.
        in_feat_output_strides (Tuple[int]): The strides of each input
            feature map, i.e., the ratio between shape of original
            image input of backbone and feature map input of FARHead.
            Default: (4, 8, 16, 32).
        out_feat_output_stride (int): The stride of output feature map.
            Default: 4.
        max_step (int): Max step iteration for calculating scale value
            in annealing function. Default: 10000.
        ignore_index (int | None): The label index to be ignored.
            Default: 255.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Default: 2.0.
        annealing_type (str): The type of annealing function.
            Default: 'cosine'.
    """

    def __init__(self,
                 in_channels=256,
                 out_channels=128,
                 in_feat_output_strides=(4, 8, 16, 32),
                 out_feat_output_stride=4,
                 max_step=10000,
                 ignore_index=255,
                 gamma=2.0,
                 annealing_type='cosine',
                 **kwargs):
        super(FARHead, self).__init__(
            in_channels=in_channels, channels=out_channels, **kwargs)
        self.in_feat_output_strides = in_feat_output_strides
        self.out_feat_output_stride = out_feat_output_stride
        self.blocks = nn.ModuleList()
        self.num_samples = []

        # These parameters are used for FarSeg loss calculation
        self.buffer_step = 0
        self.max_step = max_step
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.annealing_type = annealing_type
        for i, in_feat_os in enumerate(in_feat_output_strides):
            num_upsample = int(math.log2(int(in_feat_os))) - int(
                math.log2(int(out_feat_output_stride)))
            self.num_samples.append(num_upsample)

            num_layers = num_upsample if num_upsample != 0 else 1

            self.blocks.append(
                nn.Sequential(*[
                    nn.Sequential(
                        ConvModule(
                            in_channels=in_channels if idx ==
                            0 else out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg),
                        nn.UpsamplingBilinear2d(scale_factor=2)
                        if num_upsample != 0 else nn.Identity(),
                    ) for idx in range(num_layers)
                ]))

    def forward(self, inputs):
        self.buffer_step += 1
        for i, stride in enumerate(self.in_feat_output_strides):
            assert (stride /
                    self.out_feat_output_stride == inputs[0].shape[-1] /
                    inputs[i].shape[-1]), 'Input Feature map \
                    stride ratio must be the same of backbone!'

        inner_feat_list = []
        for i, block in enumerate(self.blocks):
            decoder_feat = block(inputs[i])
            inner_feat_list.append(decoder_feat)
        out_feats = sum(inner_feat_list) / 4.
        output = self.cls_seg(out_feats)
        return output

    def cosine_annealing(self, lower_bound, upper_bound, _t, _t_max):
        return upper_bound + 0.5 * (lower_bound - upper_bound) * (
            math.cos(math.pi * _t / _t_max) + 1)

    def poly_annealing(self, lower_bound, upper_bound, _t, _t_max):
        factor = (1 - _t / _t_max)**0.9
        return upper_bound + factor * (lower_bound - upper_bound)

    def linear_annealing(self, lower_bound, upper_bound, _t, _t_max):
        factor = 1 - _t / _t_max
        return upper_bound + factor * (lower_bound - upper_bound)

    def annealing_softmax_focalloss(self,
                                    y_pred,
                                    y_true,
                                    t,
                                    t_max,
                                    ignore_index=255,
                                    gamma=2.0,
                                    annealing_function=cosine_annealing):
        losses = cross_entropy(
            y_pred, y_true, ignore_index=ignore_index, reduction='none')
        with torch.no_grad():
            p = y_pred.softmax(dim=1)
            modulating_factor = (1 - p).pow(gamma)
            valid_mask = ~y_true.eq(ignore_index)
            masked_y_true = torch.where(valid_mask, y_true,
                                        torch.zeros_like(y_true))
            modulating_factor = torch.gather(
                modulating_factor, dim=1,
                index=masked_y_true.unsqueeze(dim=1)).squeeze_(dim=1)
            normalizer = losses.sum() / (losses * modulating_factor).sum()
            scales = modulating_factor * normalizer
        if t > t_max:
            scale = scales
        else:
            scale = annealing_function(1, scales, t, t_max)
        losses = (losses * scale).sum() / (valid_mask.sum() + p.size(0))
        return losses

    def losses(self, seg_logit, seg_label):
        """Compute annealing softmax focalloss."""
        losses = dict()
        seg_label = seg_label.float()
        func_dict = dict(
            cosine=self.cosine_annealing,
            poly=self.poly_annealing,
            linear=self.linear_annealing)
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.size()[2:],
            mode='bilinear',
            align_corners=False)
        seg_label = seg_label.squeeze(1)
        loss = self.annealing_softmax_focalloss(seg_logit, seg_label.long(),
                                                self.buffer_step,
                                                self.max_step,
                                                self.ignore_index, self.gamma,
                                                func_dict[self.annealing_type])
        losses['loss_asfocal'] = loss

        return losses
