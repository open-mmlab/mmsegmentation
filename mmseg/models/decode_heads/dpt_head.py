import math

import torch
import torch.nn as nn
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmcv.runner import BaseModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


class ReassembleBlocks(BaseModule):
    """ViTPostProcessBlock, process cls_token in ViT backbone output and
    rearrange the feature vector to feature map.

    Args:
        in_channels (int): ViT feature channels. Default: 768.
        out_channels (List): output channels of each stage.
            Default: [96, 192, 384, 768].
        readout_type (str): Type of readout operation. Default: 'ignore'.
        start_index (int): Start index of feature vector. Default: 1.
        patch_size (int): The patch size. Default: 16.
    """

    def __init__(self,
                 in_channels=768,
                 out_channels=[96, 192, 384, 768],
                 readout_type='ignore',
                 start_index=1,
                 patch_size=16):
        super(ReassembleBlocks, self).__init__()

        assert readout_type in ['ignore', 'add', 'project']
        self.readout_type = readout_type
        self.start_index = start_index
        self.patch_size = patch_size

        self.projects = [
            ConvModule(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
            ) for out_channel in out_channels
        ]

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        if self.readout_type == 'project':
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels), nn.GELU()))

    def forward(self, inputs, img_size):
        for i, x in enumerate(inputs):
            if self.readout_type == 'ignore':
                x = x[:, self.start_index:]
            elif self.readout_type == 'add':
                x = x[:, self.start_index:] + x[:, 0].unsqueeze(1)
            else:
                readout = x[:, 0].unsqueeze(1).expand_as(x[:,
                                                           self.start_index:])
                x = torch.cat((x[:, self.start_index:], readout), -1)
                x = self.readout_projects[i](x)
            B, _, C = x.shape
            x = x.reshape(B, img_size[0] // self.patch_size,
                          img_size[1] // self.patch_size,
                          C).permute(0, 3, 1, 2)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            inputs[i] = x
        return inputs


class PreActResidualConvUnit(BaseModule):
    """ResidualConvUnit, pre-activate residual unit."""

    def __init__(self,
                 in_channels,
                 act_cfg,
                 norm_cfg,
                 conv_cfg=None,
                 stride=1,
                 dilation=1,
                 init_cfg=None):
        super(PreActResidualConvUnit, self).__init__(init_cfg)

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, in_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, in_channels, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            in_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, in_channels, in_channels, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.activate = build_activation_layer(act_cfg)

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, inputs):
        x = self.activate(inputs)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv2(x)
        x = self.norm2(x)

        return x + inputs


class FeatureFusionBlock(BaseModule):
    """FeatureFusionBlock, merge feature map from different stage.

    Args:
        in_channels (int): Input channels.
        act_cfg (dict): The activation config for ResidualConvUnit.
        norm_cfg (dict): Config dict for normalization layer.
        expand (bool): Whether expand the channels in post process block.
            Default: False.
        align_corners (bool): align_corner setting for bilinear upsample.
            Default: True.
    """

    def __init__(self,
                 in_channels,
                 act_cfg,
                 norm_cfg,
                 expand=False,
                 align_corners=True):
        super(FeatureFusionBlock, self).__init__()

        self.in_channels = in_channels
        self.expand = expand
        self.align_corners = align_corners

        self.out_channels = in_channels
        if self.expand:
            self.out_channels = in_channels // 2

        self.project = ConvModule(
            self.in_channels, self.out_channels, kernel_size=1)

        self.res_conv_unit1 = PreActResidualConvUnit(
            in_channels=self.in_channels, act_cfg=act_cfg, norm_cfg=norm_cfg)
        self.res_conv_unit2 = PreActResidualConvUnit(
            in_channels=self.in_channels, act_cfg=act_cfg, norm_cfg=norm_cfg)

    def forward(self, *inputs):
        x = inputs[0]
        if len(inputs) == 2:
            x = x + self.res_conv_unit1(inputs[1])
        x = self.res_conv_unit2(x)
        x = resize(
            x,
            scale_factor=2,
            mode='bilinear',
            align_corners=self.align_corners)
        return self.project(x)


@HEADS.register_module()
class DPTHead(BaseDecodeHead):
    """Vision Transformers for Dense Prediction.

    This head is implemented of `DPT <https://arxiv.org/abs/2103.13413>`_.

    Args:
        embed_dims (int): The embed dimension of the ViT backbone.
        post_process_channels (List): Out channels of post process conv
            layers. Default: [96, 192, 384, 768].
        readout_type (str): Type of readout operation. Default: 'ignore'.
        patch_start_index (int): Start index of feature vector.
        patch_size (int): The patch size. Default: 16.
        expand_channels (bool): Whether expand the channels in post process
            block. Default: False.
        act_cfg (dict): The activation config for residual conv unit.
            Defalut 'ReLU'.
        norm_cfg (dict): Config dict for normalization layer. Default 'BN'.
    """

    def __init__(self,
                 embed_dims=768,
                 post_process_channels=[96, 192, 384, 768],
                 readout_type='ignore',
                 patch_start_index=1,
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
                                                  readout_type,
                                                  patch_start_index,
                                                  patch_size)

        self.post_process_channels = [
            channel * math.pow(2, i) if expand_channels else channel
            for i, channel in enumerate(post_process_channels)
        ]
        self.convs = nn.ModuleList()
        for _, channel in enumerate(self.post_process_channels):
            self.convs.append(
                ConvModule(channel, self.channels, kernel_size=3, padding=1))

        self.fusion_blocks = nn.ModuleList()
        for _ in range(len(self.convs)):
            self.fusion_blocks.append(
                FeatureFusionBlock(self.channels, act_cfg, norm_cfg))

        self.project = ConvModule(
            self.channels, self.channels, kernel_size=3, padding=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        x, img_size = [i[0] for i in x], x[0][1]
        x = self.reassemble_blocks(x, img_size)
        x = [self.convs[i](feature) for i, feature in enumerate(x)]

        out = self.fusion_blocks[3](x[3])
        for i in range(2, -1, -1):
            out = self.fusion_blocks[i](out, x[i])
        out = self.project(out)
        out = self.cls_seg(out)
        return out
