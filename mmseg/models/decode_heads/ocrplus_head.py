import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import DepthwiseSeparableConvModule, resize
from ..builder import HEADS
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from .cascade_decode_head import BaseCascadeDecodeHead
from .ocr_head import SpatialGatherModule, ObjectAttentionBlock


class DepthwiseSeparableObjectAttentionBlock(_SelfAttentionBlock):
    """We replace the original 1x1 conv with a separable 3x3 conv within the self.bottleneck."""

    def __init__(self, in_channels, channels, scale, conv_cfg, norm_cfg,
                 act_cfg):
        if scale > 1:
            query_downsample = nn.MaxPool2d(kernel_size=scale)
        else:
            query_downsample = None
        super(DepthwiseSeparableObjectAttentionBlock, self).__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=query_downsample,
            key_downsample=None,
            key_query_num_convs=2,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.bottleneck = DepthwiseSeparableConvModule(
            in_channels * 2,
            in_channels,
            3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, query_feats, key_feats):
        """Forward function."""
        context = super(DepthwiseSeparableObjectAttentionBlock,
                        self).forward(query_feats, key_feats)
        output = self.bottleneck(torch.cat([context, query_feats], dim=1))
        if self.query_downsample is not None:
            output = resize(query_feats)

        return output


@HEADS.register_module()
class DepthwiseSeparableOCRPlusHead(BaseCascadeDecodeHead):
    """Object-Contextual Representations for Semantic Segmentation.

    This head is augment the original `OCRNet
    <https://arxiv.org/abs/1909.11065>` with a decoder head.

    We make 3 modifications based on the OCRHead:
    -1- apply a decoder head to combine the 2x-resolution feature maps from Res-2 stage following the DeepLabv3+
    -2- replace the 3x3 conv with separable 3x3 conv that is used decrease the channel from 2048->512 (self.bottleneck)
    -3- replace the ObjectAttentionBlock with DepthwiseSeparableObjectAttentionBlock

    Args:
        ocr_channels (int): The intermediate channels of OCR block.
        c1_in_channels (int): The input channels of c1 decoder. If is 0, the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
        scale (int): The scale of probability map in SpatialGatherModule in Default: 1.
    """

    def __init__(self, ocr_channels, c1_in_channels, c1_channels, scale=1, **kwargs):
        super(DepthwiseSeparableOCRPlusHead, self).__init__(**kwargs)
        assert c1_in_channels >= 0

        self.ocr_channels = ocr_channels
        self.scale = scale
        self.object_context_block = DepthwiseSeparableObjectAttentionBlock(
            self.channels,
            self.ocr_channels,
            self.scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
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
