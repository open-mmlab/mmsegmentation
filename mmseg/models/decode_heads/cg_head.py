from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class CGHead(BaseDecodeHead):

    def __init__(self, **kwargs):
        super(CGHead, self).__init__(**kwargs)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        output = self.cls_seg(x)
        return output
