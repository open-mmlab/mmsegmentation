# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.models.decode_heads.fcn_head import FCNHead
from ..builder import HEADS


@HEADS.register_module()
class SegmenterLinearHead(FCNHead):

    def __init__(self, in_channels, **kwargs):
        kwargs['num_convs'] = 0
        super(SegmenterLinearHead, self).__init__(
            in_channels=in_channels, **kwargs)
