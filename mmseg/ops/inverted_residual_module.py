from mmcv.cnn import ConvModule, build_norm_layer
from torch import nn


class InvertedResidual(nn.Module):
    """Inverted Residual Module
    Args:
        inp (int): input channels.
        oup (int): output channels.
        stride (int): downsampling factor.
        expand_ratio (int): 1 or 2.
        dilation (int): Dilated conv. Default: 1.
        conv_cfg (dict | None): Config of conv layers. Default: None.
        norm_cfg (dict | None): Config of norm layers. Default:
            dict(type='BN').
        act_cfg (dict): Config of activation layers. Default:
            dict(type='ReLU6').
    """

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
