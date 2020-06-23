import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class FPNHead(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        start_level (int): Index of the start input feature level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input feature level (inclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        concat_all_levels (bool): If true, use fpn_conv and concat all level
            feature map together for prediction. Otherwise, the first
            lateral will be used. Default: False.
    """

    def __init__(self,
                 start_level=0,
                 end_level=-1,
                 concat_all_levels=False,
                 **kwargs):
        super(FPNHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        assert start_level < len(self.in_channels)
        if end_level < 0:
            end_level = len(self.in_channels) + end_level
        assert end_level < len(self.in_channels)
        assert len(self.in_channels) >= end_level - start_level + 1

        self.start_level = start_level
        self.end_level = end_level
        self.concat_all_levels = concat_all_levels
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.end_level + 1):
            l_conv = ConvModule(
                self.in_channels[i],
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)

        if self.concat_all_levels:
            for i in range(self.start_level, self.end_level + 1):
                fpn_conv = ConvModule(
                    self.channels,
                    self.channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False)
                self.fpn_convs.append(fpn_conv)

            self.fpn_bottleneck = ConvModule(
                len(self.in_channels) * self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, inputs):

        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        if self.concat_all_levels:
            # build outputs
            fpn_outs = [
                self.fpn_convs[i](laterals[i])
                for i in range(used_backbone_levels - 1)
            ]

            for i in range(used_backbone_levels - 1, 0, -1):
                fpn_outs[i] = resize(
                    fpn_outs[i],
                    size=fpn_outs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
            fpn_outs = torch.cat(fpn_outs, dim=1)
            output = self.fpn_bottleneck(fpn_outs)
        else:
            output = laterals[0]
        output = self.cls_seg(output)
        return output
