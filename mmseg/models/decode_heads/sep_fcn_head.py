from mmseg.ops import DepthwiseSeparableConvModule
from ..builder import HEADS
from .fcn_head import FCNHead


@HEADS.register_module()
class SepFCNHead(FCNHead):

    def __init__(self, **kwargs):
        super(SepFCNHead, self).__init__(**kwargs)
        self.convs[0] = DepthwiseSeparableConvModule(
            self.in_channels,
            self.channels,
            norm_cfg=self.norm_cfg,
            relu_first=False)
        for i in range(1, self.num_convs):
            self.convs[i] = DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                norm_cfg=self.norm_cfg,
                relu_first=False)

        if self.concat_input:
            self.conv_cat = DepthwiseSeparableConvModule(
                self.in_channels + self.channels,
                self.channels,
                self.channels,
                norm_cfg=self.norm_cfg,
                relu_first=False)
