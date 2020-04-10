import torch

from mmseg.ops import ContextBlock
from ..registry import HEADS
from .fcn_head import FCNHead


@HEADS.register_module
class GCHead(FCNHead):
    """GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond.

        This head is the implementation of:
        - Global Context block in (https://arxiv.org/abs/1904.11492)
    """

    def __init__(self, ratio=1 / 4., **kwargs):
        super(GCHead, self).__init__(num_convs=2, **kwargs)
        self.ratio = ratio
        self.gc_block = ContextBlock(self.channels, ratio)

    def forward(self, inputs):
        x = inputs[self.in_index]
        output = self.convs[0](x)
        output = self.gc_block(output)
        output = self.convs[1](output)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output
