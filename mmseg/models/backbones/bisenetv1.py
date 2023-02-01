# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmseg.registry import MODELS
from ..utils import resize


class SpatialPath(BaseModule):
    """Spatial Path to preserve the spatial size of the original input image
    and encode affluent spatial information.

    Args:
        in_channels(int): The number of channels of input
            image. Default: 3.
        num_channels (Tuple[int]): The number of channels of
            each layers in Spatial Path.
            Default: (64, 64, 64, 128).
    Returns:
        x (torch.Tensor): Feature map for Feature Fusion Module.
    """

    def __init__(self,
                 in_channels=3,
                 num_channels=(64, 64, 64, 128),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert len(num_channels) == 4, 'Length of input channels \
                                        of Spatial Path must be 4!'

        self.layers = []
        for i in range(len(num_channels)):
            layer_name = f'layer{i + 1}'
            self.layers.append(layer_name)
            if i == 0:
                self.add_module(
                    layer_name,
                    ConvModule(
                        in_channels=in_channels,
                        out_channels=num_channels[i],
                        kernel_size=7,
                        stride=2,
                        padding=3,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
            elif i == len(num_channels) - 1:
                self.add_module(
                    layer_name,
                    ConvModule(
                        in_channels=num_channels[i - 1],
                        out_channels=num_channels[i],
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
            else:
                self.add_module(
                    layer_name,
                    ConvModule(
                        in_channels=num_channels[i - 1],
                        out_channels=num_channels[i],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))

    def forward(self, x):
        for i, layer_name in enumerate(self.layers):
            layer_stage = getattr(self, layer_name)
            x = layer_stage(x)
        return x


class AttentionRefinementModule(BaseModule):
    """Attention Refinement Module (ARM) to refine the features of each stage.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
    Returns:
        x_out (torch.Tensor): Feature map of Attention Refinement Module.
    """

    def __init__(self,
                 in_channels,
                 out_channel,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.conv_layer = ConvModule(
            in_channels=in_channels,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.atten_conv_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None), nn.Sigmoid())

    def forward(self, x):
        x = self.conv_layer(x)
        x_atten = self.atten_conv_layer(x)
        x_out = x * x_atten
        return x_out


class ContextPath(BaseModule):
    """Context Path to provide sufficient receptive field.

    Args:
        backbone_cfg:(dict): Config of backbone of
            Context Path.
        context_channels (Tuple[int]): The number of channel numbers
            of various modules in Context Path.
            Default: (128, 256, 512).
        align_corners (bool, optional): The align_corners argument of
            resize operation. Default: False.
    Returns:
        x_16_up, x_32_up (torch.Tensor, torch.Tensor): Two feature maps
            undergoing upsampling from 1/16 and 1/32 downsampling
            feature maps. These two feature maps are used for Feature
            Fusion Module and Auxiliary Head.
    """

    def __init__(self,
                 backbone_cfg,
                 context_channels=(128, 256, 512),
                 align_corners=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert len(context_channels) == 3, 'Length of input channels \
                                           of Context Path must be 3!'

        self.backbone = MODELS.build(backbone_cfg)

        self.align_corners = align_corners
        self.arm16 = AttentionRefinementModule(context_channels[1],
                                               context_channels[0])
        self.arm32 = AttentionRefinementModule(context_channels[2],
                                               context_channels[0])
        self.conv_head32 = ConvModule(
            in_channels=context_channels[0],
            out_channels=context_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv_head16 = ConvModule(
            in_channels=context_channels[0],
            out_channels=context_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.gap_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(
                in_channels=context_channels[2],
                out_channels=context_channels[0],
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

    def forward(self, x):
        x_4, x_8, x_16, x_32 = self.backbone(x)
        x_gap = self.gap_conv(x_32)

        x_32_arm = self.arm32(x_32)
        x_32_sum = x_32_arm + x_gap
        x_32_up = resize(input=x_32_sum, size=x_16.shape[2:], mode='nearest')
        x_32_up = self.conv_head32(x_32_up)

        x_16_arm = self.arm16(x_16)
        x_16_sum = x_16_arm + x_32_up
        x_16_up = resize(input=x_16_sum, size=x_8.shape[2:], mode='nearest')
        x_16_up = self.conv_head16(x_16_up)

        return x_16_up, x_32_up


class FeatureFusionModule(BaseModule):
    """Feature Fusion Module to fuse low level output feature of Spatial Path
    and high level output feature of Context Path.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
    Returns:
        x_out (torch.Tensor): Feature map of Feature Fusion Module.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_atten = nn.Sequential(
            ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg), nn.Sigmoid())

    def forward(self, x_sp, x_cp):
        x_concat = torch.cat([x_sp, x_cp], dim=1)
        x_fuse = self.conv1(x_concat)
        x_atten = self.gap(x_fuse)
        # Note: No BN and more 1x1 conv in paper.
        x_atten = self.conv_atten(x_atten)
        x_atten = x_fuse * x_atten
        x_out = x_atten + x_fuse
        return x_out


@MODELS.register_module()
class BiSeNetV1(BaseModule):
    """BiSeNetV1 backbone.

    This backbone is the implementation of `BiSeNet: Bilateral
    Segmentation Network for Real-time Semantic
    Segmentation <https://arxiv.org/abs/1808.00897>`_.

    Args:
        backbone_cfg:(dict): Config of backbone of
            Context Path.
        in_channels (int): The number of channels of input
            image. Default: 3.
        spatial_channels (Tuple[int]): Size of channel numbers of
            various layers in Spatial Path.
            Default: (64, 64, 64, 128).
        context_channels (Tuple[int]): Size of channel numbers of
            various modules in Context Path.
            Default: (128, 256, 512).
        out_indices (Tuple[int] | int, optional): Output from which stages.
            Default: (0, 1, 2).
        align_corners (bool, optional): The align_corners argument of
            resize operation in Bilateral Guided Aggregation Layer.
            Default: False.
        out_channels(int): The number of channels of output.
            It must be the same with `in_channels` of decode_head.
            Default: 256.
    """

    def __init__(self,
                 backbone_cfg,
                 in_channels=3,
                 spatial_channels=(64, 64, 64, 128),
                 context_channels=(128, 256, 512),
                 out_indices=(0, 1, 2),
                 align_corners=False,
                 out_channels=256,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)
        assert len(spatial_channels) == 4, 'Length of input channels \
                                           of Spatial Path must be 4!'

        assert len(context_channels) == 3, 'Length of input channels \
                                           of Context Path must be 3!'

        self.out_indices = out_indices
        self.align_corners = align_corners
        self.context_path = ContextPath(backbone_cfg, context_channels,
                                        self.align_corners)
        self.spatial_path = SpatialPath(in_channels, spatial_channels)
        self.ffm = FeatureFusionModule(context_channels[1], out_channels)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

    def forward(self, x):
        # stole refactoring code from Coin Cheung, thanks
        x_context8, x_context16 = self.context_path(x)
        x_spatial = self.spatial_path(x)
        x_fuse = self.ffm(x_spatial, x_context8)

        outs = [x_fuse, x_context8, x_context16]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)
