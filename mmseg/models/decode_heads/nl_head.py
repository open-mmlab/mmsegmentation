import torch

from mmseg.ops import NonLocal2D
from ..registry import HEADS
from .fcn_head import FCNHead


@HEADS.register_module
class NLHead(FCNHead):
    """Non-local Neural Networks.

        This head is the implementation of:
        - Nonlocal block in (https://arxiv.org/abs/1711.07971)
    """

    def __init__(self, reduction=2, **kwargs):
        super(NLHead, self).__init__(num_convs=2, **kwargs)
        self.reduction = reduction
        self.nl_block = NonLocal2D(self.channels, reduction)

    def forward(self, inputs):
        x = inputs[self.in_index]
        output = self.convs[0](x)
        output = self.nl_block(output)
        output = self.convs[1](output)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output
