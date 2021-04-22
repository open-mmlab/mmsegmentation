import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d

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
        assert len(scales) == num_outs
        self.scales = scales
        self.num_outs = num_outs
        self.convs = [
            Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            for _ in range(num_outs)
        ]

    def forward(self, x):
        outs = []
        print(len(self.convs))
        for i in range(self.num_outs):
            scale = self.scales[i]
            x = self.convs[i](x)
            outs.append(
                F.interpolate(x, size=x.shape[:2] * scale, mode='bilinear'))
        return tuple(outs)
