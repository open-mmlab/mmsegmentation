import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class LRASPPHead(BaseDecodeHead):
    """Lite R-ASPP (LRASPP) head is proposed in Searching for MobileNetV3.

    This head is the implementation of `MobileNetV3
    <https://ieeexplore.ieee.org/document/9008835>`_.
    """

    def __init__(self, **kwargs):
        super(LRASPPHead, self).__init__(**kwargs)
        self.conv = ConvModule(
            self.in_channels[1],
            self.channels,
            1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.image_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=49, stride=(16, 20)),
            ConvModule(
                self.in_channels[1],
                self.channels,
                1,
                act_cfg=dict(type='Sigmoid')))

        self.auxiliary_cls_seg = nn.Conv2d(
            self.in_channels[0], self.num_classes, kernel_size=1)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)

        x[1] = self.conv(x[1]) * resize(
            self.image_pool(x[1]), size=x[1].size()[2:], mode='bilinear')

        x[1] = resize(x[1], size=x[0].size()[2:], mode='bilinear')

        output = self.cls_seg(x[1]) + self.auxiliary_cls_seg(x[0])

        return output
