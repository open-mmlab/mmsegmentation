from mmcv.cnn import (ConvModule, build_norm_layer, constant_init,
                      kaiming_init, normal_init)
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm
from torch import nn

from mmseg.utils import get_root_logger
from ..builder import BACKBONES


class InvertedResidual(nn.Module):

    def __init__(self,
                 inp,
                 oup,
                 stride,
                 expand_ratio,
                 dilation=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6')):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(
                ConvModule(
                    inp,
                    hidden_dim,
                    kernel_size=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        layers.extend([
            # dw
            ConvModule(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=dilation,
                stride=stride,
                dilation=dilation,
                groups=hidden_dim,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            build_norm_layer(norm_cfg, oup)[1],
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


@BACKBONES.register_module()
class MobileNetV2(nn.Module):
    arch_settings = (
        InvertedResidual,
        [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ])

    def __init__(self,
                 in_channels=3,
                 dilations=(1, 1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 input_channels=32,
                 width_mult=1.0,
                 round_nearest=8,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6')):
        """
        MobileNet V2 main class
        Args:
            width_mult (float): Width multiplier - adjusts number of channels
                in each layer by this amount
            round_nearest (int): Round the number of channels in each layer to
                be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for
                mobilenet
        """
        super(MobileNetV2, self).__init__()
        self.in_channels = in_channels
        self.width_mult = width_mult
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        block, inverted_residual_setting = self.arch_settings
        self.dilations = dilations
        self.out_indices = out_indices

        # building first layer
        input_channels = int(
            input_channels *
            self.width_mult) if self.width_mult > 1.0 else input_channels
        # last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280
        self.conv1 = ConvModule(
            3,
            input_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # building inverted residual blocks
        self.planes = input_channels
        self.block1 = self._make_layer(block, self.planes,
                                       inverted_residual_setting[0:1],
                                       dilations[0])
        self.block2 = self._make_layer(block, self.planes,
                                       inverted_residual_setting[1:2],
                                       dilations[1])
        self.block3 = self._make_layer(block, self.planes,
                                       inverted_residual_setting[2:3],
                                       dilations[2])
        self.block4 = self._make_layer(block, self.planes,
                                       inverted_residual_setting[3:5],
                                       dilations[3])
        self.block5 = self._make_layer(block, self.planes,
                                       inverted_residual_setting[5:],
                                       dilations[4])

    def _make_layer(self,
                    block,
                    planes,
                    inverted_residual_setting,
                    dilation=1):
        features = list()
        for t, c, n, s in inverted_residual_setting:
            out_channels = int(c * self.width_mult)
            stride = s if dilation == 1 else 1
            features.append(
                block(
                    planes,
                    out_channels,
                    stride,
                    t,
                    dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            planes = out_channels
            for i in range(n - 1):
                features.append(
                    block(
                        planes,
                        out_channels,
                        1,
                        t,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                planes = out_channels
        self.planes = planes
        return nn.Sequential(*features)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_out')
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, 0, 0.01)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        c1 = self.block2(x)
        c2 = self.block3(c1)
        c3 = self.block4(c2)
        c4 = self.block5(c3)

        outs = [c1, c2, c3, c4]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)
