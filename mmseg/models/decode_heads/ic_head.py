import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import BaseModule

from mmseg.core import add_prefix
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


class CascadeFeatureFusion(BaseModule):
    """CFF Unit for ICNet."""

    def __init__(self,
                 low_channels,
                 high_channels,
                 out_channels,
                 nclass,
                 conv_cfg,
                 norm_cfg,
                 act_cfg,
                 align_corners=False,
                 init_cfg=None):
        super(CascadeFeatureFusion, self).__init__(init_cfg=init_cfg)
        self.align_corners = align_corners
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
            x_low,
            size=x_high.size()[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high
        x = F.relu(x, inplace=True)
        x_low_cls = self.conv_low_cls(x_low)

        return x, x_low_cls


@HEADS.register_module()
class ICHead(BaseDecodeHead):
    """ICNet for Real-Time Semantic Segmentation on High-Resolution Images.

    This head is the implementation of ICHead in
    (https://arxiv.org/abs/1704.08545)
    """

    def __init__(self, **kwargs):
        super(ICHead, self).__init__(**kwargs)

        self.cff_24 = CascadeFeatureFusion(
            self.in_channels[2],
            self.in_channels[1],
            self.channels,
            self.num_classes,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)

        self.cff_12 = CascadeFeatureFusion(
            self.channels,
            self.in_channels[0],
            self.channels,
            self.num_classes,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        x_sub1, x_sub2, x_sub4 = inputs
        outputs = list()
        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)
        outputs.append(x_24_cls)
        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x_sub1)
        outputs.append(x_12_cls)

        up_x2 = resize(
            x_cff_12,
            scale_factor=2,
            mode='bilinear',
            align_corners=self.align_corners)
        up_x2 = self.cls_seg(up_x2)
        outputs.append(up_x2)
        up_x8 = resize(
            up_x2,
            scale_factor=4,
            mode='bilinear',
            align_corners=self.align_corners)
        outputs.append(up_x8)
        # 1 -> 1/4 -> 1/8 -> 1/16
        outputs.reverse()
        return outputs

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, only ``final`` prediction is used."""
        return self.forward(inputs)[0]

    def losses(self, seg_logit, seg_label):
        loss_names = ['final', '4x', 'cascade_1', 'cascade_2']
        loss_weights = [1.0, 0.4, 0.4, 0.4]
        loss = dict()
        for i in range(len(seg_logit)):
            loss_tmp = super(ICHead, self).losses(seg_logit[i], seg_label)
            loss_tmp['loss_seg'] *= loss_weights[i]
            loss.update(add_prefix(loss_tmp, loss_names[i]))

        return loss
