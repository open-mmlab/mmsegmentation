from mmcv.cnn import ConvModule

from ..builder import HEADS
from .fcn_head import FCNHead


@HEADS.register_module()
class FCNDilateHead(FCNHead):
    """FCN Dilate 6.

    This head is implemented of `<https://arxiv.org/abs/1911.05722>`_.

    Args:
        dilation (int): Spacing between kernel elements. Default: 6.
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
    """

    def __init__(self, dilation=6, **kwargs):
        super().__init__(**kwargs)
        assert dilation > 0 and isinstance(dilation, int)
        self.convs[0] = ConvModule(
            self.in_channels,
            self.channels,
            kernel_size=self.kernel_size,
            padding=dilation,
            norm_cfg=self.norm_cfg,
            dilation=dilation)
        for i in range(1, self.num_convs):
            self.convs[i] = ConvModule(
                self.channels,
                self.channels,
                kernel_size=self.kernel_size,
                padding=dilation,
                norm_cfg=self.norm_cfg,
                dilation=dilation)
