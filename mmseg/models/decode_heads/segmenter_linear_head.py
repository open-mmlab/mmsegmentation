import torch.nn as nn
from einops import rearrange
from mmcv.cnn.utils.weight_init import trunc_normal_

from ..builder import HEADS
from .decode_head import BaseDecodeHead


def init_weights(m, std=0.02):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=std)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


@HEADS.register_module()
class SegmenterLinearHead(BaseDecodeHead):

    def __init__(self, in_channels, init_std=0.02, **kwargs):
        super(SegmenterLinearHead, self).__init__(
            in_channels=in_channels, **kwargs)
        self.head = nn.Linear(in_channels, self.num_classes)
        self.apply(lambda x: init_weights(x, std=init_std))

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        GS = x.shape[-1]
        x = rearrange(x, 'b n h w -> b (h w) n')

        x = self.head(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=GS)

        return x
