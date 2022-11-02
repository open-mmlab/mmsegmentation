# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import DepthwiseSeparableConvModule

from mmseg.registry import MODELS
from .fcn_head import FCNHead


@MODELS.register_module()
class DepthwiseSeparableFCNHead(FCNHead):
    """Depthwise-Separable Fully Convolutional Network for Semantic
    Segmentation.

    This head is implemented according to `Fast-SCNN: Fast Semantic
    Segmentation Network <https://arxiv.org/abs/1902.04502>`_.

    Args:
        in_channels(int): Number of output channels of FFM.
        channels(int): Number of middle-stage channels in the decode head.
        concat_input(bool): Whether to concatenate original decode input into
            the result of several consecutive convolution layers.
            Default: True.
        num_classes(int): Used to determine the dimension of
            final prediction tensor.
        in_index(int): Correspond with 'out_indices' in FastSCNN backbone.
        norm_cfg (dict | None): Config of norm layers.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        loss_decode(dict): Config of loss type and some
            relevant additional options.
        dw_act_cfg (dict):Activation config of depthwise ConvModule. If it is
            'default', it will be the same as `act_cfg`. Default: None.
    """

    def __init__(self, dw_act_cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.convs[0] = DepthwiseSeparableConvModule(
            self.in_channels,
            self.channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            norm_cfg=self.norm_cfg,
            dw_act_cfg=dw_act_cfg)

        for i in range(1, self.num_convs):
            self.convs[i] = DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                norm_cfg=self.norm_cfg,
                dw_act_cfg=dw_act_cfg)

        if self.concat_input:
            self.conv_cat = DepthwiseSeparableConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                norm_cfg=self.norm_cfg,
                dw_act_cfg=dw_act_cfg)
