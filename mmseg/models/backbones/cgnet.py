import torch
import torch.nn as nn
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      kaiming_init)

from ..builder import BACKBONES


class FGlo(nn.Module):
    """Global Context Extractor for CGNet.

    This class is employed to refine the joint feature of both local feature
    and surrounding context.
    """

    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):
        num_batch, num_channel = x.size()[:2]
        y = self.avg_pool(x).view(num_batch, num_channel)
        y = self.fc(y).view(num_batch, num_channel, 1, 1)
        return x * y


class ContextGuidedBlock(nn.Module):
    """Context Guided Block for CGNet.

    This class consists of four components: local feature extractor,
    surrounding feature extractor, joint feature extractor and global
    context extractor.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        dilation (int): Dilation rate for surrounding context extractor.
        reduction (int): Reductions for global context extractor.
        conv_cfg (dict): Dictionary to construct and config conv layer.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        act_cfg (dict): Dictionary to construct and config act layer.
        add (bool): Add input to output or not.
        down (bool): Downsample the input to 1/2 or not.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation=2,
                 reduction=16,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='PReLU'),
                 add=True,
                 down=False):
        super(ContextGuidedBlock, self).__init__()
        self.down = down

        channels = out_channels if down else out_channels // 2
        if 'type' in act_cfg and act_cfg['type'] == 'PReLU':
            act_cfg['num_parameters'] = channels
        kernel_size = 3 if down else 1
        stride = 2 if down else 1
        padding = (kernel_size - 1) // 2

        self.conv1x1 = ConvModule(
            in_channels,
            channels,
            kernel_size,
            stride,
            padding,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.F_loc = build_conv_layer(
            conv_cfg,
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False)
        self.F_sur = build_conv_layer(
            conv_cfg,
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            groups=channels,
            dilation=dilation,
            bias=False)

        self.bn = build_norm_layer(norm_cfg, 2 * channels)[1]
        self.activate = nn.PReLU(2 * channels)

        if down:
            self.bottleneck = build_conv_layer(
                conv_cfg,
                2 * channels,
                out_channels,
                kernel_size=1,
                bias=False)

        self.add = add and not down
        self.F_glo = FGlo(out_channels, reduction)

    def forward(self, x):
        output = self.conv1x1(x)
        loc = self.F_loc(output)
        sur = self.F_sur(output)

        joi_feat = torch.cat([loc, sur], 1)  # the joint feature
        joi_feat = self.bn(joi_feat)
        joi_feat = self.activate(joi_feat)
        if self.down:
            joi_feat = self.bottleneck(joi_feat)  # channel = out_channels

        # F_glo is employed to refine the joint feature
        output = self.F_glo(joi_feat)

        if self.add:
            output = x + output

        return output


class InputInjection(nn.Module):
    """Downsampling module for CGNet."""

    def __init__(self, downsamplingRatio):
        super(InputInjection, self).__init__()
        self.pool = nn.ModuleList()
        for i in range(0, downsamplingRatio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)
        return x


@BACKBONES.register_module()
class CGNet(nn.Module):
    """CGNet backbone.

    A Light-weight Context Guided Network for Semantic Segmentation
    arXiv: https://arxiv.org/abs/1811.08201

    Args:
        in_channels (int): Number of input image channels. Normally 3.
        num_channels (list[int]): Numbers of feature channels at each stages.
        num_blocks (list[int]): Numbers of CG blocks at stage 1 and stage 2.
        dilation (list[int]): Dilation rate for surrounding context extractors
            at stage 1 and stage 2.
        reduction (list[int]): Reductions for global context extractors at
            stage 1 and stage 2.
        conv_cfg (dict): Dictionary to construct and config conv layer.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        act_cfg (dict): Dictionary to construct and config act layer.
    """

    def __init__(self,
                 in_channels=3,
                 num_channels=[32, 64, 128],
                 num_blocks=[3, 21],
                 dilation=[2, 4],
                 reduction=[8, 16],
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='PReLU')):

        super(CGNet, self).__init__()
        _act_cfg = act_cfg
        if _act_cfg['type'] == 'PReLU':
            _act_cfg['num_parameters'] = num_channels[0]

        cur_channels = in_channels
        self.stem = nn.ModuleList()
        for i in range(3):
            self.stem.append(
                ConvModule(
                    cur_channels,
                    num_channels[0],
                    3,
                    2 if i == 0 else 1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            cur_channels = num_channels[0]

        self.inject_2x = InputInjection(1)  # down-sample for Input, factor=2
        self.inject_4x = InputInjection(2)  # down-sample for Input, factor=4

        cur_channels += in_channels
        self.bn_prelu_0 = nn.Sequential(
            build_norm_layer(norm_cfg, cur_channels)[1],
            nn.PReLU(cur_channels))

        # stage 1
        self.level1 = nn.ModuleList()
        for i in range(0, num_blocks[0]):
            self.level1.append(
                ContextGuidedBlock(
                    cur_channels if i == 0 else num_channels[1],
                    num_channels[1],
                    dilation[0],
                    reduction[0],
                    conv_cfg,
                    norm_cfg,
                    act_cfg,
                    down=(i == 0)))  # CG block

        cur_channels = 2 * num_channels[1] + in_channels
        self.bn_prelu_1 = nn.Sequential(
            build_norm_layer(norm_cfg, cur_channels)[1],
            nn.PReLU(cur_channels))

        # stage 2
        self.level2 = nn.ModuleList()
        for i in range(0, num_blocks[1]):
            self.level2.append(
                ContextGuidedBlock(
                    cur_channels if i == 0 else num_channels[2],
                    num_channels[2],
                    dilation[1],
                    reduction[1],
                    conv_cfg,
                    norm_cfg,
                    act_cfg,
                    down=(i == 0)))  # CG block

        cur_channels = 2 * num_channels[2]
        self.bn_prelu_2 = nn.Sequential(
            build_norm_layer(norm_cfg, cur_channels)[1],
            nn.PReLU(cur_channels))

    def init_weights(self, pretrained=None):
        # init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                kaiming_init(m)
            elif classname.find('ConvTranspose2d') != -1:
                kaiming_init(m)

    def forward(self, x):
        output = []

        # stage 0
        inp_2x = self.inject_2x(x)
        inp_4x = self.inject_4x(x)
        for layer in self.stem:
            x = layer(x)
        x = self.bn_prelu_0(torch.cat([x, inp_2x], 1))
        output.append(x)

        # stage 1
        for i, layer in enumerate(self.level1):
            x = layer(x)
            if i == 0:
                down1 = x
        x = self.bn_prelu_1(torch.cat([x, down1, inp_4x], 1))
        output.append(x)

        # stage 2
        for i, layer in enumerate(self.level2):
            x = layer(x)
            if i == 0:
                down2 = x
        x = self.bn_prelu_2(torch.cat([down2, x], 1))
        output.append(x)

        return output
