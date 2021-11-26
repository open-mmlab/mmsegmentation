from ..builder import HEADS
from .fcn_head import FCNHead


@HEADS.register_module()
class SegmenterLinearHead(FCNHead):

    def __init__(self, **kwargs):
        super(SegmenterLinearHead, self).__init__(**kwargs)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        x = self.convs(x)
        return x
