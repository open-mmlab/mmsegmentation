import torch.nn as nn
from einops import rearrange
from mmcv.cnn.utils.weight_init import trunc_normal_

from ..builder import HEADS
from .fcn_head import FCNHead


@HEADS.register_module()
class SegmenterLinearHead(FCNHead):
    def __init__(self, **kwargs):
        super(SegmenterLinearHead, self).__init__(num_convs=1, kernel_size=1, **kwargs)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        x = self.convs(x)
        return x
