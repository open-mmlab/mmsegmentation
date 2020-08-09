from mmseg.ops import DepthwiseSeparableConvModule
from ..builder import HEADS
from .fcn_head import FCNHead


@HEADS.register_module()
class SepFCNHead(FCNHead):
    """Depthwise-Separable Fully Convolutional Network for Semantic
    Segmentation.

    This head is implemented according to Fast-SCNN.
    Args:
        in_channels(int): Number of output channels of FFM.

        channels(int): Number of middle-stage channels in the decode head.

        concat_input(bool): Whether to concatenate original decode input into
            the result of consecutive convolution layers.

        num_classes(int): Used to determine the dimension of
            final prediction tensor.

        in_index(int): Correspond with 'out_indices' in FastSCNN backbone.

        norm_cfg (dict|None): Config of norm layers.

        align_corners (bool): align_corners argument of F.interpolate.

        loss_decode(dict): Config of loss type and some
            relevant additional options.
    """

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
