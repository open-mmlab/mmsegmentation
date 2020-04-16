import torch
import torch.nn.functional as F

from mmseg.ops import ConvModule
from ..registry import HEADS
from .decode_head import DecodeHead


@HEADS.register_module
class ConcatHead(DecodeHead):

    def __init__(self, in_channels, channels, kernel_size=1, **kwargs):
        if isinstance(in_channels, (list, tuple)):
            in_channels = sum(in_channels)
        if isinstance(channels, (list, tuple)):
            channels = sum(channels)
        super(ConcatHead, self).__init__(
            in_channels=in_channels, channels=channels, **kwargs)
        self.kernel_size = kernel_size
        self.bottleneck = ConvModule(
            self.in_channels,
            self.channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        upsampled_inputs = [
            F.interpolate(
                input=x,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=True) for x in inputs
        ]
        output = torch.cat(upsampled_inputs, dim=1)
        output = self.bottleneck(output)
        output = self.cls_seg(output)

        return output
