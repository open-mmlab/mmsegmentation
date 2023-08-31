# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmseg.models import EncoderDecoder
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


@MODELS.register_module(name='InferExampleHead')
class ExampleDecodeHead(BaseDecodeHead):

    def __init__(self, num_classes=19, out_channels=None):
        super().__init__(
            3, 3, num_classes=num_classes, out_channels=out_channels)

    def forward(self, inputs):
        return self.cls_seg(inputs[0])


@MODELS.register_module(name='InferExampleBackbone')
class ExampleBackbone(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3)

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        return [self.conv(x)]


@MODELS.register_module(name='InferExampleModel')
class ExampleModel(EncoderDecoder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
