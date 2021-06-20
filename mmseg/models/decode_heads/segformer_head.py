import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@HEADS.register_module()
class SegFormerHead(BaseDecodeHead):
    """The MLP Head of segformer."""

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super(SegFormerHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.interpolate_mode = interpolate_mode

        embed_dim = self.channels

        self.conv_c4 = ConvModule(
            in_channels=self.in_channels[-1],
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.conv_c3 = ConvModule(
            in_channels=self.in_channels[-2],
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.conv_c2 = ConvModule(
            in_channels=self.in_channels[-3],
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.conv_c1 = ConvModule(
            in_channels=self.in_channels[-4],
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv_fuse = ConvModule(
            in_channels=embed_dim * 4,
            out_channels=embed_dim,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        x = self._transform_inputs(inputs)
        c1, c2, c3, c4 = x

        # MLP decoder on C1-C4
        n, _, h, w = c1.shape

        out_c4 = self.conv_c4(c4)
        out_c4 = F.interpolate(
            out_c4,
            size=(h, w),
            mode=self.mode,
            align_corners=self.align_corners)

        out_c3 = self.conv_c3(c3)
        out_c3 = F.interpolate(
            out_c3,
            size=(h, w),
            mode=self.mode,
            align_corners=self.align_corners)

        out_c2 = self.conv_c2(c2)
        out_c2 = F.interpolate(
            out_c2,
            size=(h, w),
            mode=self.mode,
            align_corners=self.align_corners)

        out_c1 = self.conv_c1(c1)

        out = self.conv_fuse(
            torch.cat([out_c4, out_c3, out_c2, out_c1], dim=1))

        x = self.cls_seg(out)

        return x
