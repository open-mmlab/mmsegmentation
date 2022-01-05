import math
import torch
import torch.nn as nn

from mmcv.cnn import  ConvModule
from mmcv.runner.base_module import BaseModule, ModuleList
from mmseg.ops import resize

from ..builder import HEADS
from .decode_head import BaseDecodeHead

class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, norm_cfg, act_cfg):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.conv1 = ConvModule(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            order=("act", "conv", "norm")
        )
        self.conv2 =ConvModule(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            order=("act", "conv", "norm")
        )
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(self, features, groups=1, expand=False, norm_cfg=None, act_cfg=dict(type='ReLU'), align_corners=True):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()
        self.align_corners = align_corners
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.expand = expand
        self.groups = groups

        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = ConvModule(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )#parameter bias and groups  should be adjusted

        self.resConfUnit1 = ResidualConvUnit(features, self.norm_cfg, self.act_cfg)
        self.resConfUnit2 = ResidualConvUnit(features, self.norm_cfg, self.act_cfg)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        
        output = resize(
                output,
                scale_factor=2,
                mode='bilinear',
                align_corners=self.align_corners)
        output = self.out_conv(output)

        return output

@HEADS.register_module()
class DPTHead(BaseDecodeHead):
    """ Dense Prediction Transformer Head 
    A PyTorch implement of : `Vision Transformers for Dense Prediction`
        https://arxiv.org/abs/2103.13413
        
    Inspiration from
        https://github.com/intel-isl/DPT

    Args:

    """
    def __init__(self, groups=1, expand=False, **kwargs):
        super(DPTHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.groups = groups
        self.expand = expand
        # Simulated FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for index, in_channel in enumerate(self.in_channels):
            l_conv = ConvModule(
                in_channel,
                self.channels*(1<<index if self.expand else 1),
                kernel_size=3,
                stride=1,
                padding=1,
                groups=self.groups,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = FeatureFusionBlock(
                self.channels,
                groups=self.groups,
                expand=self.expand,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                align_corners=self.align_corners)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            self.channels,
            self.channels,
            3,
            padding=1,
            bias=False,
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
        x = self.fpn_convs[-1](laterals[-1])
        for i in range(used_backbone_levels - 1, 0, -1):
            x = self.fpn_convs[i](x,laterals[i-1])

        output = self.fpn_bottleneck(x)
        output = self.cls_seg(output)
    
        return output


            self.in_channels,
            self.out_channels,
            kernel_size=1,
            act_cfg=None,
            bias=True)

        self.res_conv_unit1 = PreActResidualConvUnit(
            in_channels=self.in_channels, act_cfg=act_cfg, norm_cfg=norm_cfg)
        self.res_conv_unit2 = PreActResidualConvUnit(
            in_channels=self.in_channels, act_cfg=act_cfg, norm_cfg=norm_cfg)

    def forward(self, *inputs):
        x = inputs[0]
        if len(inputs) == 2:
            if x.shape != inputs[1].shape:
                res = resize(
                    inputs[1],
                    size=(x.shape[2], x.shape[3]),
                    mode='bilinear',
                    align_corners=False)
            else:
                res = inputs[1]
            x = x + self.res_conv_unit1(res)
        x = self.res_conv_unit2(x)
        x = resize(
            x,
            scale_factor=2,
            mode='bilinear',
            align_corners=self.align_corners)
        x = self.project(x)
        return x


@HEADS.register_module()
class DPTHead(BaseDecodeHead):
    """Vision Transformers for Dense Prediction.

    This head is implemented of `DPT <https://arxiv.org/abs/2103.13413>`_.

    Args:
        embed_dims (int): The embed dimension of the ViT backbone.
            Default: 768.
        post_process_channels (List): Out channels of post process conv
            layers. Default: [96, 192, 384, 768].
        readout_type (str): Type of readout operation. Default: 'ignore'.
        patch_size (int): The patch size. Default: 16.
        expand_channels (bool): Whether expand the channels in post process
            block. Default: False.
        act_cfg (dict): The activation config for residual conv unit.
            Default dict(type='ReLU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
    """

    def __init__(self,
                 embed_dims=768,
                 post_process_channels=[96, 192, 384, 768],
                 readout_type='ignore',
                 patch_size=16,
                 expand_channels=False,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN'),
                 **kwargs):
        super(DPTHead, self).__init__(**kwargs)

        self.in_channels = self.in_channels
        self.expand_channels = expand_channels
        self.reassemble_blocks = ReassembleBlocks(embed_dims,
                                                  post_process_channels,
                                                  readout_type, patch_size)

        self.post_process_channels = [
            channel * math.pow(2, i) if expand_channels else channel
            for i, channel in enumerate(post_process_channels)
        ]
        self.convs = nn.ModuleList()
        for channel in self.post_process_channels:
            self.convs.append(
                ConvModule(
                    channel,
                    self.channels,
                    kernel_size=3,
                    padding=1,
                    act_cfg=None,
                    bias=False))
        self.fusion_blocks = nn.ModuleList()
        for _ in range(len(self.convs)):
            self.fusion_blocks.append(
                FeatureFusionBlock(self.channels, act_cfg, norm_cfg))
        self.fusion_blocks[0].res_conv_unit1 = None
        self.project = ConvModule(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg)
        self.num_fusion_blocks = len(self.fusion_blocks)
        self.num_reassemble_blocks = len(self.reassemble_blocks.resize_layers)
        self.num_post_process_channels = len(self.post_process_channels)
        assert self.num_fusion_blocks == self.num_reassemble_blocks
        assert self.num_reassemble_blocks == self.num_post_process_channels

    def forward(self, inputs):
        assert len(inputs) == self.num_reassemble_blocks
        x = self._transform_inputs(inputs)
        x = self.reassemble_blocks(x)
        x = [self.convs[i](feature) for i, feature in enumerate(x)]
        out = self.fusion_blocks[0](x[-1])
        for i in range(1, len(self.fusion_blocks)):
            out = self.fusion_blocks[i](out, x[-(i + 1)])
        out = self.project(out)
        out = self.cls_seg(out)
        return out
