import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from .cascade_decode_head import BaseCascadeDecodeHead
from .ocr_head import SpatialGatherModule, ObjectAttentionBlock


@HEADS.register_module()
class OCRPlusHead(BaseCascadeDecodeHead):
    """Object-Contextual Representations for Semantic Segmentation.

    This head is augment the original `OCRNet
    <https://arxiv.org/abs/1909.11065>` with a decoder head.

    Args:
        ocr_channels (int): The intermediate channels of OCR block.
        scale (int): The scale of probability map in SpatialGatherModule in
            Default: 1.
    """

    def __init__(self, ocr_channels, c1_in_channels, c1_channels, scale=1, **kwargs):
        super(OCRPlusHead, self).__init__(**kwargs)
        assert c1_in_channels >= 0

        self.ocr_channels = ocr_channels
        self.scale = scale
        self.object_context_block = ObjectAttentionBlock(
            self.channels,
            self.ocr_channels,
            self.scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.spatial_gather_module = SpatialGatherModule(self.scale)

        self.bottleneck = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.fuse_bottleneck = nn.Sequential(
            ConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.c1_bottleneck = None


    def forward(self, inputs, prev_output):
        """Forward function."""
        x = self._transform_inputs(inputs)
        feats = self.bottleneck(x)
        context = self.spatial_gather_module(feats, prev_output)
        output = self.object_context_block(feats, context)

        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)

        output = self.fuse_bottleneck(output)
        output = self.cls_seg(output)

        return output
