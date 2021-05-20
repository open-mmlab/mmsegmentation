import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead


class MLAModule(nn.ModuleList):
    """Multi level feature aggregation Module.

    Args:
        align_corners (bool): Whether to use align_corners of F.interpolate.
        aggregation_stages (int): The number of aggregation input.
        in_channels (list): MLA module input channels.
            Default: [256, 256, 256, 256].
        mla_channels (int): MLA module output channels. Default: 128.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
    """

    def __init__(self,
                 align_corners,
                 aggregation_stages,
                 in_channels=[256, 256, 256, 256],
                 mla_channels=128,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super(MLAModule, self).__init__()
        self.align_corners = align_corners
        for i in range(aggregation_stages):
            self.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=in_channels[i],
                        out_channels=mla_channels,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg),
                    ConvModule(
                        in_channels=mla_channels,
                        out_channels=mla_channels,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg)))

    def forward(self, inputs):
        outs = []
        for x, op in zip(inputs, self):
            out = F.interpolate(
                op(x), (4 * x.shape[-2], 4 * x.shape[-1]),
                mode='bilinear',
                align_corners=self.align_corners)
            outs.append(out)
        return torch.cat(outs, dim=1)


@HEADS.register_module()
class SETRMLAHead(BaseDecodeHead):
    """Multi level feature aggretation head of SETR.

    MLA head of `SETR  <https://arxiv.org/pdf/2012.15840.pdf>`.

    Args:
        mlahead_channels (int): Channels of conv-conv-4x of multi-level feature
            aggregation. Default: 128.
    """

    def __init__(self, mla_channels=128, **kwargs):
        super(SETRMLAHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.mla_channels = mla_channels

        num_inputs = len(self.in_channels)

        # Refer to self.cls_seg settings of BaseDecodeHead
        assert self.channels == num_inputs * mla_channels

        self.mlahead = MLAModule(
            align_corners=self.align_corners,
            aggregation_stages=num_inputs,
            in_channels=self.in_channels,
            mla_channels=self.mla_channels,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        out = self.mlahead(inputs)
        out = self.cls_seg(out)
        return out
