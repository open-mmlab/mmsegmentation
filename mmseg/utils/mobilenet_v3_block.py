import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer
from torch import nn


class SqueezeBlock(nn.module):

    def __init__(self, hidden_dim, divide=4):
        super(SqueezeBlock, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // divide), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // divide, hidden_dim),
            build_activation_layer(dict(type='HSigma')))

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = F.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1)
        return out * x


class MobileNetV3Block(nn.module):
    """"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 kernel_size,
                 expand_ratio,
                 is_SE=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='HSwish')):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_SE = is_SE
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        hidden_dim = int(round(in_channels * expand_ratio))
        self.input_pointwise_conv = ConvModule(
            in_channels,
            hidden_dim,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.depthwise_conv = ConvModule(
            hidden_dim,
            hidden_dim,
            stride=stride,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.squeeze_block = SqueezeBlock(hidden_dim, divide=4)
        self.output_pointwise_conv = ConvModule(
            hidden_dim,
            out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        out = self.input_pointwise_conv(x)
        out = self.depthwise_conv(out)

        if self.is_SE:
            out = self.squeeze_block(out)

        out = self.output_pointwise_conv(out)

        if self.use_res_connect:
            return x + out
        else:
            return out
