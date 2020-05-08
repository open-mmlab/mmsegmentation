from mmcv.cnn import build_norm_layer
from torch import nn


class SeparableConvModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 relu_first=True,
                 bias=False,
                 norm_cfg=dict(type='BN')):
        super(SeparableConvModule, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=in_channels,
            bias=bias)
        self.norm_depth_name, norm_depth = build_norm_layer(
            norm_cfg, in_channels, postfix='_depth')
        self.add_module(self.norm_depth_name, norm_depth)

        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.norm_point_name, norm_point = build_norm_layer(
            norm_cfg, out_channels, postfix='_point')
        self.add_module(self.norm_point_name, norm_point)

        self.relu_first = relu_first
        self.relu = nn.ReLU(inplace=not relu_first)

    @property
    def norm_depth(self):
        return getattr(self, self.norm_depth_name)

    @property
    def norm_point(self):
        return getattr(self, self.norm_point_name)

    def forward(self, x):
        if self.relu_first:
            out = self.relu(x)
            out = self.depthwise(out)
            out = self.norm_depth(out)
            out = self.pointwise(out)
            out = self.norm_point(out)
        else:
            out = self.depthwise(x)
            out = self.norm_depth(out)
            out = self.relu(out)
            out = self.pointwise(out)
            out = self.norm_point(out)
            out = self.relu(out)
        return out
