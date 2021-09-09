# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

from mmseg.ops import resize
from ..builder import BACKBONES, build_backbone


class STDCModule(BaseModule):
    """STDCModule.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        stride (int): Stride for the first conv layer.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): The activation config for conv layers.
        num_convs (int): Numbers of conv layers.
        fusion_type (str): Type of fusion operation. Default: 'add'.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 norm_cfg=None,
                 act_cfg=None,
                 num_convs=4,
                 fusion_type='add',
                 init_cfg=None):
        super().__init__(init_cfg)
        assert num_convs > 1
        assert fusion_type in ['add', 'cat']
        self.stride = stride
        self.with_avg_pool = True if stride == 2 else False
        self.fusion_type = fusion_type

        self.layers = ModuleList()
        conv_0 = ConvModule(
            in_channels, out_channels // 2, kernel_size=1, norm_cfg=norm_cfg)

        if self.with_avg_pool:
            self.avg_pool = ConvModule(
                out_channels // 2,
                out_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=out_channels // 2,
                norm_cfg=norm_cfg,
                act_cfg=None)

            if self.fusion_type == 'add':
                self.layers.append(nn.Sequential(conv_0, self.avg_pool))
                self.skip = Sequential(
                    ConvModule(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=in_channels,
                        norm_cfg=norm_cfg,
                        act_cfg=None),
                    ConvModule(
                        in_channels,
                        out_channels,
                        1,
                        norm_cfg=norm_cfg,
                        act_cfg=None))
            else:
                self.layers.append(conv_0)
                self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.layers.append(conv_0)

        for i in range(1, num_convs):
            out_factor = 2**(i + 1) if i != num_convs - 1 else 2**i
            self.layers.append(
                ConvModule(
                    out_channels // 2**i,
                    out_channels // out_factor,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, inputs):
        if self.fusion_type == 'add':
            out = self.forward_add(inputs)
        else:
            out = self.forward_cat(inputs)
        return out

    def forward_add(self, inputs):
        layer_outputs = []
        x = inputs.clone()
        for layer in self.layers:
            x = layer(x)
            layer_outputs.append(x)
        if self.with_avg_pool:
            inputs = self.skip(inputs)

        return torch.cat(layer_outputs, dim=1) + inputs

    def forward_cat(self, inputs):
        x0 = self.layers[0](inputs)
        layer_outputs = [x0]
        for i, layer in enumerate(self.layers[1:]):
            if i == 0:
                if self.with_avg_pool:
                    x = layer(self.avg_pool(x0))
                else:
                    x = layer(x0)
            else:
                x = layer(x)
            layer_outputs.append(x)
        if self.with_avg_pool:
            layer_outputs[0] = self.skip(x0)
        return torch.cat(layer_outputs, dim=1)


class AttentionRefinementModule(BaseModule):
    """Attention Refinement Module.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): The activation config for conv layers.
            Default: dict(type='Sigmoid').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='Sigmoid'),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.conv = ConvModule(
            in_channels, out_channels, 3, padding=1, norm_cfg=norm_cfg)
        self.conv_attn = ConvModule(
            out_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x = self.conv(x)
        attn = self.conv_attn(F.avg_pool2d(x, x.shape[2:]))
        return torch.mul(x, attn)


class FeatureFusionModule(BaseModule):
    """Feature Fusion Module.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        scale_factor (int): Channel scale factor, Default: 4.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        default_act_cfg (dict): The activation config for conv layers.
            Default: dict(type='ReLU').
        act_cfg (dict): The activation config for conv layers.
            Default: dict(type='Sigmoid').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 norm_cfg=dict(type='BN'),
                 default_act_cfg=dict(type='ReLU'),
                 act_cfg=dict(type='Sigmoid'),
                 init_cfg=None):
        super().__init__(init_cfg)
        channels = out_channels // scale_factor
        self.conv0 = ConvModule(
            in_channels,
            out_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=default_act_cfg)
        self.conv1 = ConvModule(
            out_channels,
            channels,
            1,
            norm_cfg=None,
            bias=False,
            act_cfg=default_act_cfg)
        self.conv2 = ConvModule(
            channels,
            out_channels,
            1,
            norm_cfg=None,
            bias=False,
            act_cfg=act_cfg)

    def forward(self, spatial_inputs, context_inputs):
        inputs = torch.cat([spatial_inputs, context_inputs], dim=1)
        x = self.conv0(inputs)
        attn = F.avg_pool2d(x, x.shape[2:])
        attn = self.conv1(attn)
        attn = self.conv2(attn)
        x_attn = torch.mul(x, attn)
        return x_attn + x


@BACKBONES.register_module()
class StdcNet(BaseModule):
    """ STDC Net
    A PyTorch implement of : `Rethinking BiSeNet For Real-time Semantic
    Segmentation` - https://arxiv.org/abs/2104.13188

    Modified from:
        https://github.com/MichaelFan01/STDC-Seg

    Args:
        depth (int): The type of arch.
        in_channels (int): The num of input_channels.
        channels (tuple[int]): The output channels for each stage.
        bottleneck_type (str): The type of STDC Module type, the value must
            be 'add' or 'cat'.
        norm_cfg (dict): Config dict for normalization layer.
        act_cfg (dict): The activation config for conv layers.
        stdc_num_convs (int): Numbers of conv layer at each STDC Module.
            Default: 4.
        with_final_conv (bool): Whether add a conv layer at the Module output.
            Default: True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
        pretrained (str, optional): Model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    arch_settings = {
        'STDCNet813': [(2, 1), (2, 1), (2, 1)],
        'STDCNet1446': [(2, 1, 1, 1), (2, 1, 1, 1, 1), (2, 1, 1)]
    }

    def __init__(self,
                 stdc_type,
                 in_channels,
                 channels,
                 bottleneck_type,
                 norm_cfg,
                 act_cfg,
                 stdc_num_convs=4,
                 with_final_conv=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if stdc_type not in self.arch_settings:
            raise KeyError(f'invalid depth {stdc_type} for stdcnet')

        assert bottleneck_type in ['add', 'cat'],\
            f'bottleneck_type must be `add` or `cat`, got {bottleneck_type}'

        assert len(channels) == 5

        self.in_channels = in_channels
        self.channels = channels
        self.stage_strides = self.arch_settings[stdc_type]
        self.prtrained = pretrained
        self.stdc_num_convs = stdc_num_convs
        self.with_final_conv = with_final_conv
        self.with_cp = with_cp

        self.stages = ModuleList([
            ConvModule(
                self.in_channels,
                self.channels[0],
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                self.channels[0],
                self.channels[1],
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        ])

        for strides in self.stage_strides:
            idx = len(self.stages) - 1
            self.stages.append(
                self._make_stage(self.channels[idx], self.channels[idx + 1],
                                 strides, norm_cfg, act_cfg, bottleneck_type))

        if self.with_final_conv:
            self.final_conv = ConvModule(
                self.channels[-1],
                max(1024, self.channels[-1]),
                1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def _make_stage(self, in_channels, out_channels, strides, norm_cfg,
                    act_cfg, bottleneck_type):
        layers = []
        for i, stride in enumerate(strides):
            layers.append(
                STDCModule(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    stride,
                    norm_cfg,
                    act_cfg,
                    num_convs=self.stdc_num_convs,
                    fusion_type=bottleneck_type))
        return Sequential(*layers)

    def forward(self, x):
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        if self.with_final_conv:
            outs[-1] = self.final_conv(outs[-1])
        # feature maps shape: (x2, x4, x8, x16, x32)
        # channels will be match with the parameter `channels`
        return tuple(outs)


@BACKBONES.register_module()
class STDCContextPathNet(BaseModule):
    """STDCNet with Context Path.

    Args:
        stdc_cfg (dict): Config dict for stdc backbone.
        in_channels (tuple(int)), The last two feature maps' channels from
            stdc backbone. Default: (512, 1024).
        out_channels (int): Channels of output feature maps. Default: 128.
        ffm_cfg (dict): Config dict for FeatureFusionModule. Default:
            dict(in_channels=512, out_channels=256, scale_factor=4)
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        upsample_mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        align_corners (str): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 stdc_cfg,
                 in_channels=(512, 1024),
                 out_channels=128,
                 ffm_cfg=dict(
                     in_channels=512, out_channels=256, scale_factor=4),
                 norm_cfg=dict(type='BN'),
                 upsample_mode='nearest',
                 align_corners=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.backbone = build_backbone(stdc_cfg)
        self.arms = ModuleList()
        self.convs = ModuleList()
        for channels in in_channels:
            self.arms.append(AttentionRefinementModule(channels, out_channels))
            self.convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    norm_cfg=norm_cfg))
        self.conv_avg = ConvModule(
            in_channels[-1], out_channels, 1, norm_cfg=norm_cfg)

        self.ffm = FeatureFusionModule(**ffm_cfg)

        self.upsample_mode = upsample_mode
        self.align_corners = align_corners

    def forward(self, x):
        outs = list(self.backbone(x))
        prev_stages_out = outs[:3]
        outs = outs[2:]
        # We only use the last three layers' output from STDCNet to
        # encode context info.
        assert len(outs) == 3
        avg = F.avg_pool2d(outs[-1], outs[-1].shape[2:])
        avg = self.conv_avg(avg)

        feature_up = resize(
            avg,
            size=outs[-1].shape[2:],
            mode=self.upsample_mode,
            align_corners=self.align_corners)
        arms_out = []
        for i in range(len(self.arms) - 1, -1, -1):
            x_arm = self.arms[i](outs[i + 1]) + feature_up
            feature_up = resize(
                x_arm,
                size=outs[i].shape[2:],
                mode=self.upsample_mode,
                align_corners=self.align_corners)
            feature_up = self.convs[i](feature_up)
            arms_out.append(feature_up)
        # feature maps shape: (x2, x4, x8, x8, x8, x16)
        # the last two feature maps's channel will be equal to
        # the parameter `out_channels`
        return prev_stages_out + [self.ffm(outs[0], arms_out[1])] + list(
            reversed(arms_out))
