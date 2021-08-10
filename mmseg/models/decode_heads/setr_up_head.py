import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer

from mmseg.ops import Upsample
from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class SETRUPHead(BaseDecodeHead):
    """Naive upsampling head and Progressive upsampling head of SETR.

    Naive or PUP head of `SETR  <https://arxiv.org/pdf/2012.15840.pdf>`.

    Args:
        norm_layer (dict): Config dict for input normalization.
            Default: norm_layer=dict(type='LN', eps=1e-6, requires_grad=True).
        num_convs (int): Number of decoder convolutions. Default: 1.
        up_scale (int): The scale factor of interpolate. Default:4.
        kernel_size (int): The kernel size of convolution when decoding
            feature information from backbone. Default: 3.
        init_cfg (dict | list[dict] | None): Initialization config dict.
            Default: dict(
                     type='Constant', val=1.0, bias=0, layer='LayerNorm').
    """

    def __init__(self,
                 norm_layer=dict(type='LN', eps=1e-6, requires_grad=True),
                 num_convs=1,
                 up_scale=4,
                 kernel_size=3,
                 init_cfg=[
                     dict(type='Constant', val=1.0, bias=0, layer='LayerNorm'),
                     dict(
                         type='Normal',
                         std=0.01,
                         override=dict(name='conv_seg'))
                 ],
                 **kwargs):

        assert kernel_size in [1, 3], 'kernel_size must be 1 or 3.'

        super(SETRUPHead, self).__init__(init_cfg=init_cfg, **kwargs)

        assert isinstance(self.in_channels, int)

        _, self.norm = build_norm_layer(norm_layer, self.in_channels)

        self.up_convs = nn.ModuleList()
        in_channels = self.in_channels
        out_channels = self.channels
        for _ in range(num_convs):
            self.up_convs.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=int(kernel_size - 1) // 2,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    Upsample(
                        scale_factor=up_scale,
                        mode='bilinear',
                        align_corners=self.align_corners)))
            in_channels = out_channels

    def forward(self, x):
        x = self._transform_inputs(x)

        n, c, h, w = x.shape
        x = x.reshape(n, c, h * w).transpose(2, 1).contiguous()
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(n, c, h, w).contiguous()

        for up_conv in self.up_convs:
            x = up_conv(x)
        out = self.cls_seg(x)
        return out
