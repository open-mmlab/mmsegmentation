import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner.base_module import BaseModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM


@HEADS.register_module()
class SFNetHead(BaseDecodeHead):
    """Semantic Flow for Fast and Accurate SceneParsing.

    This head is the implementation of
    `SFSegNet <https://arxiv.org/pdf/2002.10120>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
        fpn_inplanes (list):
            The list of feature channels number from backbone.
        fpn_dim (int, optional):
            The input channels of FAM module.
            Default: 256 for ResNet50, 128 for ResNet18.
    """

    def __init__(self,
                 pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=[256, 512, 1024, 2048],
                 fpn_dim=256,
                 **kwargs):
        super(SFNetHead, self).__init__(**kwargs)
        assert isinstance(pool_scales, (list, tuple))
        self.pool_scales = pool_scales
        self.fpn_inplanes = fpn_inplanes
        self.fpn_dim = fpn_dim
        self.psp_modules = PPM(
            self.pool_scales,
            self.in_channels,
            self.in_channels // 4,
            bias=True,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=True)
        self.bottleneck = ConvModule(
            self.in_channels * 2,
            self.channels,
            3,
            padding=1,
            bias=True,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.fpn_in = []
        for fpn_inplane in self.fpn_inplanes[:-1]:
            self.fpn_in.append(
                ConvModule(
                    fpn_inplane,
                    self.fpn_dim,
                    kernel_size=1,
                    bias=True,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False))
        self.fpn_in = nn.ModuleList(self.fpn_in)
        self.fpn_out = []
        self.fpn_out_align = []
        self.dsn = []
        for i in range(len(self.fpn_inplanes) - 1):
            self.fpn_out.append(
                ConvModule(
                    self.fpn_dim,
                    self.fpn_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=True))
            self.fpn_out_align.append(
                AlignedModule(
                    inplane=self.fpn_dim, outplane=self.fpn_dim // 2))

        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)
        self.conv_last = ConvModule(
            len(self.fpn_inplanes) * self.fpn_dim,
            self.fpn_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            inplace=True)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x)[::-1])
        psp_outs = torch.cat(psp_outs, dim=1)
        psp_out = self.bottleneck(psp_outs)

        f = psp_out
        fpn_feature_list = [psp_out]

        for i in reversed(range(len(inputs) - 1)):
            conv_x = inputs[i]
            conv_x = self.fpn_in[i](conv_x)
            f = self.fpn_out_align[i]([conv_x, f])
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(
                nn.functional.interpolate(
                    fpn_feature_list[i],
                    output_size,
                    mode='bilinear',
                    align_corners=True))

        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)
        output = self.cls_seg(x)

        return output


class AlignedModule(BaseModule):
    """The implementation of Flow Alignment Module (FAM).

    Args:
       inplane (int): The number of FAM input channles.
       outplane (int): The number of FAM output channles.
    """

    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(
            outplane * 2, 2, kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = resize(
            h_feature, size=size, mode='bilinear', align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        """Implementation of Warp Procedure in Fig 3(b) of original paper,
        which is between Flow Field and High Resolution Feature Map.

        Args:
            input (Tensor): High Resolution Feature Map.
            flow (Tensor): Semantic Flow Field that will give
                dynamic indication about how to align these
                two feature maps effectively.
            size (Tuple): Shape of height and width of output.

        Returns:
            output (Tensor): High Resolution Feature Map after
                warped offset and bilinear interpolation.

        For example, in cityscapes 1024x2048 dataset with ResNet18 config,
        feature map from backbone is:
        [[1, 64, 256, 512],
        [1, 128, 128, 256],
        [1, 256, 64, 128],
        [1, 512, 32, 64]]

        Thus, its inverse shape of [input, flow, size] is:
        [[1, 128, 32, 64], [1, 2, 64, 128], (64, 128)],
        [[1, 128, 64, 128], [1, 2, 128, 256], (128, 256)], and
        [[1, 128, 128, 256], [1, 2, 256, 512], (256, 512)], respectively.

        The final output is:
        [[1, 128, 64, 128],
        [1, 128, 128, 256],
        [1, 128, 256, 512]], respectively.
        """

        out_h, out_w = size
        n, c, h, w = input.size()

        # Warped offset in grid, from -1 to 1.
        norm = torch.tensor([[[[out_w,
                                out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)

        # Warped grid which is corrected the flow offset.
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        # Sampling mechanism interpolates the values of the 4-neighbors
        # (top-left, top-right, bottom-left, and bottom-right) of input.
        output = nn.functional.grid_sample(input, grid, align_corners=True)
        return output
