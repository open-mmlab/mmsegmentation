import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer

from ..builder import HEADS
from ..utils import trunc_normal_
from .decode_head import BaseDecodeHead


class UPModule(nn.ModuleList):

    def __init__(self,
                 in_channels,
                 out_channels=256,
                 num_convs=1,
                 up_mode='bilinear',
                 up_scale=4,
                 align_corners=False,
                 num_up_layer=0,
                 kernel_size=3,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super(UPModule, self).__init__()
        self.num_convs = num_convs
        self.up_mode = up_mode
        self.up_scale = up_scale
        self.align_corners = align_corners
        self.num_up_layer = num_up_layer
        self.kernel_size = kernel_size

        for i in range(num_convs):
            if i == 0:
                tmp_in_channels = in_channels
                tmp_out_channels = out_channels
            else:
                tmp_in_channels = out_channels
                tmp_out_channels = out_channels
            self.append(
                ConvModule(
                    in_channels=tmp_in_channels,
                    out_channels=tmp_out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=int(kernel_size - 1) // 2,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, x):
        # There is an upsample defined in segmentor class..
        up_rest_count = self.num_up_layer
        for op in self:
            x = op(x)
            if up_rest_count > 0:
                x = F.interpolate(
                    x,
                    size=(x.shape[-2] * self.up_scale,
                          x.shape[-1] * self.up_scale),
                    mode=self.up_mode,
                    align_corners=self.align_corners)
                up_rest_count -= 1

        return x


@HEADS.register_module()
class SETRUPHead(BaseDecodeHead):
    """Naive upsampling head and Progressive upsampling head of SETR.

    Naive or PUP head of `SETR  <https://arxiv.org/pdf/2012.15840.pdf>`.

    Args:
        embed_dim (int): embedding dimension. Default: 1024.
        norm_layer (dict): Config dict for input normalization.
            Default: norm_layer=dict(type='LN', eps=1e-6, requires_grad=True).
        num_convs (int): Number of decoder convolutions. Default: 1.
        up_mode (str): Interpolate mode of upsampling. Default: bilinear.
        up_scale (int): The scale factor of interpolate. Default:4.
        num_up_layer (str): Nunber of upsampling layers. Default: 1.
        kernel_size (int): The kernel size of convolution when decoding
            feature information from backbone. Default: 3.
    """

    def __init__(self,
                 norm_layer=dict(type='LN', eps=1e-6, requires_grad=True),
                 num_convs=1,
                 up_mode='bilinear',
                 up_scale=4,
                 num_up_layer=0,
                 kernel_size=3,
                 **kwargs):

        assert kernel_size in [1, 3], 'kernel_size must be 1 or 3.'

        super(SETRUPHead, self).__init__(**kwargs)

        assert isinstance(self.in_channels, int)

        _, self.norm = build_norm_layer(norm_layer, self.in_channels)

        self.up = UPModule(
            in_channels=self.in_channels,
            out_channels=self.channels,
            num_convs=num_convs,
            up_mode=up_mode,
            up_scale=up_scale,
            align_corners=self.align_corners,
            num_up_layer=num_up_layer,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self._transform_inputs(x)

        n, c, h, w = x.shape
        x = x.reshape(n, c, h * w).transpose(2, 1)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(n, c, h, w)

        x = self.up(x)
        out = self.cls_seg(x)
        return out
