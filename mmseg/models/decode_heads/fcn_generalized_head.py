# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class FCNGenHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNGenHead, self).__init__(**kwargs)
        if num_convs == 0 and self.input_transform is None:
            # TODO add assert for multiple_select
            assert self.in_channels == self.channels, f"{self.in_channels} != {self.channels}"

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels[-1],
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

        for i in range(len(self.in_index)-1):
            upsample_head = []
            upsample_head.append(
                nn.ConvTranspose2d(self.num_classes, self.num_classes, (4, 4), (2, 2), padding=(1, 1)))
            if self.norm_cfg:
                upsample_head.append(build_norm_layer(
                    self.norm_cfg, self.num_classes)[1])

            m = nn.Sequential(*upsample_head)
            self.add_module(f'upsample_head{i}', m)
            c = nn.Conv2d(self.in_channels[i], self.num_classes, 1)
            self.add_module(f'cls_head{i}', c)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        x1 = x[-1] if self.input_transform == 'multiple_select' else x
        output = self.convs(x1)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x1, output], dim=1))
        output = self.cls_seg(output)
        features = x[:-1] if self.input_transform == 'multiple_select' else []
        upsample_heads = [m for name, m in self.named_children()
                          if 'upsample_head' in name]
        cls_heads = [m for name, m in self.named_children()
                     if 'cls_head' in name]
        # We keep the indices in an intuitive way in the config, so the order is actually reversed
        for inp, upsample_head, cls_head in reversed(list(zip(features, upsample_heads, cls_heads))):
            prev = upsample_head(output)
            cur = cls_head(inp)
            output = prev + cur

        return output
