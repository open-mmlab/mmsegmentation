import math
import torch
import torch.nn as nn

from mmcv.cnn import  trunc_normal_init
from mmcv.runner.base_module import BaseModule, ModuleList

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class SFHead(BaseDecodeHead):
    """ Segmenter-Linear
    A PyTorch implement of : `Segmenter: Transformer for Semantic Segmentation`
        https://arxiv.org/abs/2105.05633
        
    Inspiration from
        https://github.com/rstrudel/segmenter
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.head = nn.Linear(self.in_channels, self.num_classes)
        self.apply(init_weights)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = self.head(x)
        output = x.view(-1, H, W, self.num_classes).permute(0, 3, 1, 2).contiguous()

        return output

