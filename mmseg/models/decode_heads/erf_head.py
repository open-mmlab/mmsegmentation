# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class ERFHead(BaseDecodeHead):
    """ERFNet backbone.

    This decoder head is the implementation of `ERFNet: Efficient
    Residual Factorized ConvNet for Real-time SemanticSegmentation
    <https://ieeexplore.ieee.org/document/8063438>`_.

    Actually it is one ConvTranspose2d operation.
    """

    def __init__(self, **kwargs):
        super(ERFHead, self).__init__(**kwargs)
        self.output_conv = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.channels,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            bias=True)

    def forward(self, inputs):
        """Forward function."""
        output = self.output_conv(inputs)
        return output
