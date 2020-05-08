import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer, constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmseg.ops import SeparableConvModule
from mmseg.utils import get_root_logger
from ..builder import BACKBONES


class XceptionBlock(nn.Module):

    def __init__(self,
                 channel_list,
                 stride=1,
                 dilation=1,
                 skip_connection_type='conv',
                 relu_first=True,
                 low_feat=False,
                 with_cp=True,
                 norm_cfg=dict(type='BN')):
        super().__init__()

        assert len(channel_list) == 4
        self.skip_connection_type = skip_connection_type
        self.relu_first = relu_first
        self.low_feat = low_feat
        self.with_cp = with_cp

        if self.skip_connection_type == 'conv':
            self.conv = nn.Conv2d(
                channel_list[0],
                channel_list[-1],
                1,
                stride=stride,
                bias=False)
            self.norm_name, norm = build_norm_layer(norm_cfg, channel_list[-1])
            self.add_module(self.norm_name, norm)

        self.sep_conv1 = SeparableConvModule(
            channel_list[0],
            channel_list[1],
            dilation=dilation,
            relu_first=relu_first,
            norm_cfg=norm_cfg)
        self.sep_conv2 = SeparableConvModule(
            channel_list[1],
            channel_list[2],
            dilation=dilation,
            relu_first=relu_first,
            norm_cfg=norm_cfg)
        self.sep_conv3 = SeparableConvModule(
            channel_list[2],
            channel_list[3],
            dilation=dilation,
            relu_first=relu_first,
            stride=stride,
            norm_cfg=norm_cfg)
        self.last_inp_channels = channel_list[3]

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def forward(self, inputs):

        def _inner_forward(inputs):
            sc1 = self.sep_conv1(inputs)
            sc2 = self.sep_conv2(sc1)
            residual = self.sep_conv3(sc2)

            return residual, sc2

        if self.with_cp and inputs.requires_grad:
            residual, sc2 = cp.checkpoint(_inner_forward, inputs)
        else:
            residual, sc2 = _inner_forward(inputs)

        if self.skip_connection_type == 'conv':
            shortcut = self.conv(inputs)
            shortcut = self.norm(shortcut)
            outputs = residual + shortcut
        elif self.skip_connection_type == 'sum':
            outputs = residual + inputs
        elif self.skip_connection_type == 'none':
            outputs = residual
        else:
            raise ValueError('Unsupported skip connection type.')

        if self.low_feat:
            return outputs, sc2
        else:
            return outputs


@BACKBONES.register_module()
class Xception65(nn.Module):

    def __init__(self, output_stride, with_cp=False, norm_cfg=dict(type='BN')):
        super().__init__()
        if output_stride == 32:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 1)
            exit_block_stride = 2
        elif output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
            exit_block_stride = 1
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
            exit_block_stride = 1
        else:
            raise NotImplementedError

        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, 32, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, 64, postfix=2)
        self.add_module(self.norm2_name, norm2)

        self.block1 = XceptionBlock([64, 128, 128, 128],
                                    stride=2,
                                    norm_cfg=norm_cfg)
        self.block2 = XceptionBlock([128, 256, 256, 256],
                                    stride=2,
                                    low_feat=True,
                                    norm_cfg=norm_cfg)
        self.block3 = XceptionBlock([256, 728, 728, 728],
                                    stride=entry_block3_stride,
                                    low_feat=True,
                                    norm_cfg=norm_cfg)

        # Middle flow (16 units)
        self.block4 = XceptionBlock([728, 728, 728, 728],
                                    dilation=middle_block_dilation,
                                    skip_connection_type='sum',
                                    norm_cfg=norm_cfg)
        self.block5 = XceptionBlock([728, 728, 728, 728],
                                    dilation=middle_block_dilation,
                                    skip_connection_type='sum',
                                    norm_cfg=norm_cfg)
        self.block6 = XceptionBlock([728, 728, 728, 728],
                                    dilation=middle_block_dilation,
                                    skip_connection_type='sum',
                                    norm_cfg=norm_cfg)
        self.block7 = XceptionBlock([728, 728, 728, 728],
                                    dilation=middle_block_dilation,
                                    skip_connection_type='sum',
                                    norm_cfg=norm_cfg)
        self.block8 = XceptionBlock([728, 728, 728, 728],
                                    dilation=middle_block_dilation,
                                    skip_connection_type='sum',
                                    norm_cfg=norm_cfg)
        self.block9 = XceptionBlock([728, 728, 728, 728],
                                    dilation=middle_block_dilation,
                                    skip_connection_type='sum',
                                    norm_cfg=norm_cfg)
        self.block10 = XceptionBlock([728, 728, 728, 728],
                                     dilation=middle_block_dilation,
                                     skip_connection_type='sum',
                                     norm_cfg=norm_cfg)
        self.block11 = XceptionBlock([728, 728, 728, 728],
                                     dilation=middle_block_dilation,
                                     skip_connection_type='sum',
                                     norm_cfg=norm_cfg)
        self.block12 = XceptionBlock([728, 728, 728, 728],
                                     dilation=middle_block_dilation,
                                     skip_connection_type='sum',
                                     norm_cfg=norm_cfg)
        self.block13 = XceptionBlock([728, 728, 728, 728],
                                     dilation=middle_block_dilation,
                                     skip_connection_type='sum',
                                     norm_cfg=norm_cfg)
        self.block14 = XceptionBlock([728, 728, 728, 728],
                                     dilation=middle_block_dilation,
                                     skip_connection_type='sum',
                                     norm_cfg=norm_cfg)
        self.block15 = XceptionBlock([728, 728, 728, 728],
                                     dilation=middle_block_dilation,
                                     skip_connection_type='sum',
                                     norm_cfg=norm_cfg)
        self.block16 = XceptionBlock([728, 728, 728, 728],
                                     dilation=middle_block_dilation,
                                     skip_connection_type='sum',
                                     norm_cfg=norm_cfg)
        self.block17 = XceptionBlock([728, 728, 728, 728],
                                     dilation=middle_block_dilation,
                                     skip_connection_type='sum',
                                     norm_cfg=norm_cfg)
        self.block18 = XceptionBlock([728, 728, 728, 728],
                                     dilation=middle_block_dilation,
                                     skip_connection_type='sum',
                                     norm_cfg=norm_cfg)
        self.block19 = XceptionBlock([728, 728, 728, 728],
                                     dilation=middle_block_dilation,
                                     skip_connection_type='sum',
                                     norm_cfg=norm_cfg)

        # Exit flow
        self.block20 = XceptionBlock([728, 728, 1024, 1024],
                                     stride=exit_block_stride,
                                     dilation=exit_block_dilations[0],
                                     norm_cfg=norm_cfg)
        self.block21 = XceptionBlock([1024, 1536, 1536, 2048],
                                     dilation=exit_block_dilations[1],
                                     skip_connection_type='none',
                                     relu_first=False,
                                     norm_cfg=norm_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.block1(x)
        x, c1 = self.block2(x)  # b, h//4, w//4, 256
        x, c2 = self.block3(x)  # b, h//8, w//8, 728

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        c3 = self.block19(x)

        # Exit flow
        x = self.block20(c3)
        c4 = self.block21(x)

        return c1, c2, c3, c4


# -------------------------------------------------
#                   For DFANet
# -------------------------------------------------
class BlockA(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 norm_cfg=None,
                 start_with_relu=True):
        super(BlockA, self).__init__()
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv2d(
                in_channels, out_channels, 1, stride, bias=False)
            self.skipbn = build_norm_layer(norm_cfg, out_channels)[1]
        else:
            self.skip = None
        self.relu = nn.ReLU()
        rep = list()
        inter_channels = out_channels // 4

        if start_with_relu:
            rep.append(self.relu)
        rep.append(
            SeparableConvModule(
                in_channels, inter_channels, 3, 1, dilation,
                norm_cfg=norm_cfg))
        rep.append(norm_cfg(inter_channels))

        rep.append(self.relu)
        rep.append(
            SeparableConvModule(
                inter_channels,
                inter_channels,
                3,
                1,
                dilation,
                norm_cfg=norm_cfg))
        rep.append(norm_cfg(inter_channels))

        if stride != 1:
            rep.append(self.relu)
            rep.append(
                SeparableConvModule(
                    inter_channels, out_channels, 3, stride,
                    norm_cfg=norm_cfg))
            rep.append(norm_cfg(out_channels))
        else:
            rep.append(self.relu)
            rep.append(
                SeparableConvModule(
                    inter_channels, out_channels, 3, 1, norm_cfg=norm_cfg))
            rep.append(norm_cfg(out_channels))
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        out = self.rep(x)
        if self.skip is not None:
            skip = self.skipbn(self.skip(x))
        else:
            skip = x
        out = out + skip
        return out


class Enc(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 blocks,
                 norm_cfg=dict(type='BN')):
        super(Enc, self).__init__()
        block = list()
        block.append(BlockA(in_channels, out_channels, 2, norm_cfg=norm_cfg))
        for i in range(blocks - 1):
            block.append(
                BlockA(out_channels, out_channels, 1, norm_cfg=norm_cfg))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class FCAttention(nn.Module):

    def __init__(self, in_channels, norm_cfg=dict(type='BN')):
        super(FCAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 1000)
        self.conv = nn.Sequential(
            nn.Conv2d(1000, in_channels, 1, bias=False), norm_cfg(in_channels),
            nn.ReLU(True))

    def forward(self, x):
        n, c, _, _ = x.size()
        att = self.avgpool(x).view(n, c)
        att = self.fc(att).view(n, 1000, 1, 1)
        att = self.conv(att)
        return x * att.expand_as(x)


@BACKBONES.register_module()
class XceptionA(nn.Module):

    def __init__(self, num_classes=1000, norm_cfg=dict(type='BN')):
        super(XceptionA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, 2, 1, bias=False), norm_cfg(8), nn.ReLU(True))

        self.enc2 = Enc(8, 48, 4, norm_cfg=norm_cfg)
        self.enc3 = Enc(48, 96, 6, norm_cfg=norm_cfg)
        self.enc4 = Enc(96, 192, 4, norm_cfg=norm_cfg)

        self.fca = FCAttention(192, norm_cfg=norm_cfg)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.conv1(x)

        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.fca(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
