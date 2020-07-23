import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import DepthwiseSeparableConvModule, resize
from ..builder import HEADS
from .cascade_decode_head import BaseCascadeDecodeHead
from .ocr_head import ObjectAttentionBlock, SpatialGatherModule


@HEADS.register_module()
class OCRPlusHead(BaseCascadeDecodeHead):
    """Object-Contextual Representations for Semantic Segmentation.

    This head is a variant of the original `OCRNet
    <https://arxiv.org/abs/1909.11065>` via adding a decoder head.

    We make 3 modifications based on the OCRHead:
    -1- apply a decoder head to combine the 2x-resolution feature maps
        from Res-2 stage following the DeepLabv3+
    -2- replace the 3x3 conv -> separable 3x3 conv that is used decrease
        the channel from 2048->512 (self.bottleneck)
    -3- replace the ObjectAttentionBlock ->
        DepthwiseSeparableObjectAttentionBlock

    Args:
        ocr_channels (int): The intermediate channels of OCR block.
        c1_in_channels (int): The input channels of c1 decoder.
                              If is 0, the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
        scale (int): The scale of probability map in SpatialGatherModule.
    """
    def __init__(self,
                 ocr_channels,
                 c1_in_channels,
                 c1_channels,
                 scale=1,
                 use_sep_conv=True,
                 **kwargs):
        super(OCRPlusHead, self).__init__(**kwargs)
        assert c1_in_channels >= 0

        self.ocr_channels = ocr_channels
        self.scale = scale
        self.use_sep_conv = use_sep_conv

        self.object_context_block = ObjectAttentionBlock(
            self.channels,
            self.ocr_channels,
            self.scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            use_sep_conv=self.use_sep_conv)
        self.spatial_gather_module = SpatialGatherModule(self.scale)

        self.bottleneck = DepthwiseSeparableConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.fuse_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
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
