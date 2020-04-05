import torch

from mmseg.ops import CrissCrossAttention
from ..registry import HEADS
from .fcn_head import FCNHead


@HEADS.register_module
class CCHead(FCNHead):

    def __init__(self, recurrence=2, **kwargs):
        super(CCHead, self).__init__(num_convs=2, **kwargs)
        self.recurrence = recurrence
        self.cca = CrissCrossAttention(self.channels)

    def forward(self, inputs):
        x = inputs[self.in_index]
        output = self.convs[0](x)
        for _ in range(self.recurrence):
            self.cca(output)
        output = self.convs[1](output)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output
