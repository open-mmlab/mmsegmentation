import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class LR_ASPPHead(BaseDecodeHead):

    def __init__(self, mid_channels=128, **kwargs):
        super(LR_ASPPHead, self).__init__(**kwargs)

        higher_res_channels = self.in_channels[0]
        lower_res_channels = self.in_channels[1]

        self.mid_channels = mid_channels
        self.lower_res_conv = ConvModule(
            lower_res_channels,
            self.mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU'))

        self.attention_branch_ops = nn.Sequential(
            nn.AvgPool2d(kernel_size=49, stride=(16, 20), padding=49 // 2),
            ConvModule(
                lower_res_channels,
                self.mid_channels,
                kernel_size=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=dict(type='Sigmoid')))

        self.higher_res_conv = ConvModule(
            higher_res_channels,
            self.num_classes,
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.lower_res_final = ConvModule(
            self.mid_channels,
            self.num_classes,
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        x_higher_res = inputs[0]
        x_lower_res = inputs[1]

        # prediction from the higher-res branch.
        pred_higher_res = self.higher_res_conv(x_higher_res)

        # prediction from the lower-res branch.
        attention_vct = self.attention_branch_ops(x_lower_res)
        pointwise_attention = resize(
            attention_vct,
            size=x_lower_res.size()[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        x_lower_res_conved = self.lower_res_conv(x_lower_res)

        pred_lower_res = x_lower_res_conved * pointwise_attention
        pred_lower_res = self.lower_res_final(
            resize(
                pred_lower_res,
                size=x_higher_res.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners))

        return pred_lower_res + pred_higher_res
