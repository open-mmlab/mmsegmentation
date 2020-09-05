import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


class CascadeFeatureFusion(nn.Module):
    """CFF Unit for ICNet"""

    def __init__(self, low_channels, high_channels, out_channels, nclass,
                 conv_cfg, norm_cfg, act_cfg):
        super(CascadeFeatureFusion, self).__init__()
        self.conv_low = ConvModule(
            low_channels,
            out_channels,
            3,
            padding=2,
            dilation=2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv_high = ConvModule(
            high_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv_low_cls = build_conv_layer(
            conv_cfg, out_channels, nclass, kernel_size=1, bias=False)

    def forward(self, x_low, x_high):
        x_low = resize(
            x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high
        x = F.relu(x, inplace=True)
        x_low_cls = self.conv_low_cls(x_low)

        return x, x_low_cls


@HEADS.register_module()
class ICHead(BaseDecodeHead):
    """ICNet for Real-Time Semantic Segmentation on High-Resolution Images

        This head is the implementation of ICHead
        in (https://arxiv.org/abs/1704.08545)
    """

    def __init__(self,
                 num_channels=[64, 256, 256],
                 out_channels=[128, 128],
                 **kwargs):
        super(ICHead, self).__init__(**kwargs)

        self.cff_24 = CascadeFeatureFusion(
            num_channels[2],
            num_channels[1],
            out_channels[0],
            self.num_classes,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.cff_12 = CascadeFeatureFusion(
            out_channels[0],
            num_channels[0],
            out_channels[1],
            self.num_classes,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        # inputs = self._transform_inputs(inputs)
        x_sub1, x_sub2, x_sub4 = inputs
        outputs = list()
        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)
        outputs.append(x_24_cls)
        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x_sub1)
        outputs.append(x_12_cls)

        up_x2 = resize(
            x_cff_12, scale_factor=2, mode='bilinear', align_corners=True)
        up_x2 = self.cls_seg(up_x2)
        outputs.append(up_x2)
        up_x8 = resize(
            up_x2, scale_factor=4, mode='bilinear', align_corners=True)
        outputs.append(up_x8)
        # 1 -> 1/4 -> 1/8 -> 1/16
        outputs.reverse()
        return outputs

    def get_seg(self, inputs):
        return self.forward(inputs)[0]

    def losses(self, seg_logit, seg_label):
        loss_names = ['final', '4x', 'cascade_1', 'cascade_2']
        loss_weights = [1.0, 0.4, 0.4, 0.4]
        loss = dict()
        for i in range(4):
            loss_tmp = super(ICHead, self).losses(seg_logit[i], seg_label)
            loss_tmp['loss_seg'] *= loss_weights[i]
            loss[f'loss_seg_{loss_names[i]}'] = loss_tmp['loss_seg']
            loss.update(loss_tmp)

        return loss
