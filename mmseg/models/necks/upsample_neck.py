import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import NECKS


@NECKS.register_module()
class UpsampleNeck(nn.Module):
    """Upsample Network."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 scales=[0.5, 1, 2, 4],
                 num_outs=4):
        super(UpsampleNeck, self).__init__()
        assert isinstance(in_channels, list)
        assert len(scales) == num_outs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.num_outs = num_outs
        self.lateral_convs = nn.ModuleList()
        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            self.lateral_convs.append(
                ConvModule(in_channel, out_channels, kernel_size=1))
        for _ in range(num_outs):
            self.convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1))

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        inputs = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # support num_outs == 1 or num_outs == 4
        if len(inputs) == 1:
            inputs = inputs * self.num_outs
        outs = []
        for i in range(self.num_outs):
            x_resize = F.interpolate(
                inputs[i],
                size=list((np.array(inputs[i].shape[2:]) *
                           self.scales[i]).astype(int)),
                mode='bilinear')
            outs.append(self.convs[i](x_resize))
        return tuple(outs)
