import math

import torch
import torch.nn as nn
from mmcv.cnn import (Conv2d, ConvModule, build_activation_layer,
                      build_norm_layer)

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


class ViTPostProcessBlock(nn.Module):
    """ViTPostProcessBlock, process cls_token in ViT backbone output and resize
    the feature vector to feature map.

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
        super(ViTPostProcessBlock, self).__init__()

        assert readout_type in ['ignore', 'add', 'project']
        self.readout_type = readout_type
        self.start_index = start_index
        self.patch_size = patch_size

        self.convs = [
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

    def forward(self, inputs, img_size):
        for i, x in enumerate(inputs):
            if self.readout_type == 'ignore':
                x = x[:, self.start_index:]
            elif self.readout_type == 'add':
                x = x[:, self.start_index] + x[:, 0].unsqueeze(1)
            else:
                readout = x[:, 0].unsqueeze(1).expand_as(x[:,
                                                           self.start_index])
                x = torch.cat((x[:, self.start_indx], readout), -1)
            B, _, C = x.shape
            x = x.reshape(B, img_size[0] // self.patch_size,
                          img_size[1] // self.patch_size,
                          C).permute(0, 3, 1, 2)
            x = self.convs[i](x)
            x = self.resize_layers[i](x)
            inputs[i] = x
        return inputs


class ResidualConvUnit(nn.Module):
    """ResidualConvUnit, pre-activate residual unit.

    Args:
        in_channels (int): Input channels.
        act_cfg (dict): The activation config before conv.
        norm_cfg (dict): Config dict for normalization layer.
    """

    def __init__(self, in_channels, act_cfg, norm_cfg):
        super(ResidualConvUnit, self).__init__()
        self.channels = in_channels

        self.activation = build_activation_layer(act_cfg)
        self.bn = False if norm_cfg is None else True
        self.bias = not self.bn

        self.conv1 = ConvModule(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            bias=self.bias)

        self.conv2 = ConvModule(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            bias=self.bias)

        if self.bn:
            _, self.bn1 = build_norm_layer(norm_cfg, self.channels)
            _, self.bn2 = build_norm_layer(norm_cfg, self.channels)

    def forward(self, inputs):
        x = self.activation(inputs)
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x)

        x = self.activation(x)
        x = self.conv2(x)
        if self.bn:
            x = self.bn2(x)

        return x + inputs


class FeatureFusionBlock(nn.Module):
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

        self.out_conv = Conv2d(
            self.in_channels, self.out_channels, kernel_size=1)

        self.res_conv_unit1 = ResidualConvUnit(self.in_channels, act_cfg,
                                               norm_cfg)
        self.res_conv_unit2 = ResidualConvUnit(self.in_channels, act_cfg,
                                               norm_cfg)

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
        return self.out_conv(x)


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
                 **kwards):
        super(DPTHead, self).__init__(**kwards)

        self.in_channels = self.in_channels
        self.expand_channels = expand_channels
        self.post_process_block = ViTPostProcessBlock(embed_dims,
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
                Conv2d(channel, self.channels, kernel_size=3, padding=1))

        self.refinenet0 = FeatureFusionBlock(self.channels, act_cfg, norm_cfg)
        self.refinenet1 = FeatureFusionBlock(self.channels, act_cfg, norm_cfg)
        self.refinenet2 = FeatureFusionBlock(self.channels, act_cfg, norm_cfg)
        self.refinenet3 = FeatureFusionBlock(self.channels, act_cfg, norm_cfg)

        self.conv = ConvModule(
            self.channels, self.channels, kernel_size=3, padding=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        x, img_size = [i[0] for i in x], x[0][1]
        x = self.post_process_block(x, img_size)
        x = [self.convs[i](feature) for i, feature in enumerate(x)]

        path_3 = self.refinenet3(x[3])
        path_2 = self.refinenet2(path_3, x[2])
        path_1 = self.refinenet1(path_2, x[1])
        path_0 = self.refinenet0(path_1, x[0])
        x = self.conv(path_0)
        output = self.cls_seg(x)
        return output
