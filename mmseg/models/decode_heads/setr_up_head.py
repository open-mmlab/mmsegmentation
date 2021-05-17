import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer

from ..builder import HEADS
from ..utils import trunc_normal_
from .decode_head import BaseDecodeHead


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
        num_up_layer (str): Nunber of upsampling layers. Default: 1.
        kernel_size (int): The kernel size of convolution when decoding
            feature information from backbone. Default: 3.
    """

    def __init__(self,
                 embed_dim=1024,
                 norm_layer=dict(type='LN', eps=1e-6, requires_grad=True),
                 num_convs=2,
                 up_mode='bilinear',
                 num_up_layer=1,
                 kernel_size=3,
                 **kwargs):

        assert num_convs in [
            2, 4
        ], 'naive head (num_convs = 2), pup head (num_convs=4).'

        assert kernel_size in [1, 3], 'kernel_size must be 1 or 3.'

        super(SETRUPHead, self).__init__(**kwargs)

        assert isinstance(self.in_channels, int)
        assert embed_dim == self.in_channels

        self.num_convs = num_convs
        _, self.norm = build_norm_layer(norm_layer, embed_dim)
        self.up_mode = up_mode
        self.num_up_layer = num_up_layer
        self.kernel_size = kernel_size

        out_channel = self.num_classes

        if self.num_convs == 2:
            self.conv_0 = nn.Conv2d(
                embed_dim,
                256,
                kernel_size=kernel_size,
                stride=1,
                padding=int(kernel_size - 1) // 2)
            self.conv_seg = nn.Conv2d(
                256, out_channel, kernel_size=1, stride=1)
            _, self.unified_bn_fc_0 = build_norm_layer(self.norm_cfg, 256)

        elif self.num_convs == 4:
            self.conv_0 = nn.Conv2d(
                embed_dim, 256, kernel_size=kernel_size, stride=1, padding=1)
            self.conv_1 = nn.Conv2d(
                256, 256, kernel_size=kernel_size, stride=1, padding=1)
            self.conv_2 = nn.Conv2d(
                256, 256, kernel_size=kernel_size, stride=1, padding=1)
            self.conv_3 = nn.Conv2d(
                256, 256, kernel_size=kernel_size, stride=1, padding=1)
            self.conv_seg = nn.Conv2d(
                256, out_channel, kernel_size=1, stride=1)

            _, self.unified_bn_fc_0 = build_norm_layer(self.norm_cfg, 256)
            _, self.unified_bn_fc_1 = build_norm_layer(self.norm_cfg, 256)
            _, self.unified_bn_fc_2 = build_norm_layer(self.norm_cfg, 256)
            _, self.unified_bn_fc_3 = build_norm_layer(self.norm_cfg, 256)

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

        if self.num_convs == 2:
            if self.num_up_layer == 1:
                x = self.conv_0(x)
                x = self.unified_bn_fc_0(x)
                x = F.relu(x, inplace=True)
                x = self.conv_seg(x)
            elif self.num_up_layer == 2:
                x = self.conv_0(x)
                x = self.unified_bn_fc_0(x)
                x = F.relu(x, inplace=True)
                x = F.interpolate(
                    x,
                    size=(x.shape[-2] * 4, x.shape[-1] * 4),
                    mode=self.up_mode,
                    align_corners=self.align_corners)
                x = self.conv_seg(x)
            else:
                raise NotImplementedError
        elif self.num_convs == 4:
            if self.num_up_layer == 4:
                x = self.conv_0(x)
                x = self.unified_bn_fc_0(x)
                x = F.relu(x, inplace=True)
                x = F.interpolate(
                    x,
                    size=(x.shape[-2] * 2, x.shape[-1] * 2),
                    mode=self.up_mode,
                    align_corners=self.align_corners)
                x = self.conv_1(x)
                x = self.unified_bn_fc_1(x)
                x = F.relu(x, inplace=True)
                x = F.interpolate(
                    x,
                    size=(x.shape[-2] * 2, x.shape[-1] * 2),
                    mode=self.up_mode,
                    align_corners=self.align_corners)
                x = self.conv_2(x)
                x = self.unified_bn_fc_2(x)
                x = F.relu(x, inplace=True)
                x = F.interpolate(
                    x,
                    size=(x.shape[-2] * 2, x.shape[-1] * 2),
                    mode=self.up_mode,
                    align_corners=self.align_corners)
                x = self.conv_3(x)
                x = self.unified_bn_fc_3(x)
                x = F.relu(x, inplace=True)
                x = self.conv_seg(x)
            else:
                raise NotImplementedError

        return x
