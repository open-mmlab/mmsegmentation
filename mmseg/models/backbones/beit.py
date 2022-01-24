# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import BACKBONES
from .vit import VisionTransformer


@BACKBONES.register_module()
class BEiT(VisionTransformer):
    """VisionTransformer with support for patch."""

    def __init__(self, **kwargs):
        super(BEiT, self).__init__(**kwargs)
