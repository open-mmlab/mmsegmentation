import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class SETRMLAAUXHead(BaseDecodeHead):
    """Vision Transformer with support for patch or hybrid CNN input stage.

    The extra head of MLA head of `SETR
    <https://arxiv.org/pdf/2012.15840.pdf>`

    Args:
        img_size (tuple): Input image size. Default: (224, 224).
        mla_channels (int): Channels of reshape-conv of multi-level feature
            aggregation. Default: 256.
    """

    def __init__(self, img_size=(224, 224), mla_channels=256, **kwargs):
        super(SETRMLAAUXHead, self).__init__(**kwargs)

        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        elif isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            raise TypeError('img_size must be type of int or tuple')

        self.mla_channels = mla_channels

        if self.in_channels != self.mla_channels:
            self.conv_proj = ConvModule(
                self.in_channels,
                self.mla_channels,
                kernel_size=1,
                bias=False,
                act_cfg=None)

        self.conv_seg = ConvModule(
            self.mla_channels,
            self.num_classes,
            kernel_size=1,
            bias=False,
            act_cfg=None)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)

        if self.in_channels != self.mla_channels:
            x = self.conv_proj(x)

        x = self.conv_seg(x)
        x = F.interpolate(
            x,
            size=self.img_size,
            mode='bilinear',
            align_corners=self.align_corners)
        return x
