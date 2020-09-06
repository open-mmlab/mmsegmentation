from mmcv.cnn import ConvModule, build_norm_layer
from torch import nn


class InvertedResidual(nn.Module):
    """Inverted residual module.

    Args:
        in_channels (int): The input channels of the InvertedResidual block.
        out_channels (int): The output channels of the InvertedResidual block.
        stride (int): Stride of the middle (first) 3x3 convolution.
        expand_ratio (int): adjusts number of channels of the hidden layer
            in InvertedResidual by this amount.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expand_ratio,
                 dilation=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6')):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 \
            and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(
                ConvModule(
                    in_channels,
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
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
