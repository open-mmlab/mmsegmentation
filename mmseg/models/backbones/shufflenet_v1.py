import copy

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, build_activation_layer, constant_init,
                      normal_init)
from mmcv.runner import BaseModule, load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmseg.utils import get_root_logger
from ..builder import BACKBONES
from ..utils import channel_shuffle, make_divisible


class ShuffleUnit(BaseModule):
    """ShuffleUnit block.

    ShuffleNet unit with pointwise group convolution (GConv) and channel
    shuffle.

    Args:
        in_channels (int): The input channels of the ShuffleUnit.
        out_channels (int): The output channels of the ShuffleUnit.
        groups (int, optional): The number of groups to be used in grouped 1x1
            convolutions in each ShuffleUnit. Default: 3
        first_block (bool, optional): Whether it is the first ShuffleUnit of a
            sequential ShuffleUnits. Default: True, which means not using the
            grouped 1x1 convolution.
        combine (str, optional): The ways to combine the input and output
            branches. Default: 'add'.
        downsample (bool): Whether to downsample the feature map when combine
            is 'concat'. Default: 'True'.
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 groups=3,
                 first_block=True,
                 combine='add',
                 downsample=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False):
        super(ShuffleUnit, self).__init__()
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        act_cfg = copy.deepcopy(act_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_block = first_block
        self.combine = combine
        self.downsample = downsample
        self.groups = groups
        self.bottleneck_channels = self.out_channels // 4
        self.with_cp = with_cp

        if self.combine == 'add':
            self.depthwise_stride = 1
            self._combine_func = self._add
            assert in_channels == out_channels, (
                'in_channels must be equal to out_channels when combine '
                'is add')
        elif self.combine == 'concat':
            self._combine_func = self._concat
            self.out_channels -= self.in_channels
            if self.downsample:
                self.depthwise_stride = 2
                self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            else:
                self.depthwise_stride = 1
        else:
            raise ValueError(f'Cannot combine tensors with {self.combine}. '
                             'Only "add" and "concat" are supported')

        self.first_1x1_groups = 1 if first_block else self.groups
        self.g_conv_1x1_compress = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.bottleneck_channels,
            kernel_size=1,
            groups=self.first_1x1_groups,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.depthwise_conv3x3_bn = ConvModule(
            in_channels=self.bottleneck_channels,
            out_channels=self.bottleneck_channels,
            kernel_size=3,
            stride=self.depthwise_stride,
            padding=1,
            groups=self.bottleneck_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.g_conv_1x1_expand = ConvModule(
            in_channels=self.bottleneck_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            groups=self.groups,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.act = build_activation_layer(act_cfg)

    @staticmethod
    def _add(x, out):
        # residual connection
        return x + out

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):

        def _inner_forward(x):
            residual = x

            out = self.g_conv_1x1_compress(x)
            out = self.depthwise_conv3x3_bn(out)

            if self.groups > 1:
                out = channel_shuffle(out, self.groups)

            out = self.g_conv_1x1_expand(out)

            if self.combine == 'concat':
                if self.downsample:
                    residual = self.avgpool(residual)
                out = self.act(out)
                out = self._combine_func(residual, out)
            else:
                out = self._combine_func(residual, out)
                out = self.act(out)
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


@BACKBONES.register_module()
class ShuffleNetV1(BaseModule):
    """ShuffleNetV1 backbone.

    Args:
        groups (int, optional): The number of groups to be used in grouped 1x1
            convolutions in each ShuffleUnit. Default: 3.
        widen_factor (float, optional): Width multiplier - adjusts the number
            of channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (0, 1, 2).
        stage_blocks (Sequence[int]): The number of blocks in each stage.
            Default: (4, 8, 4).
        downsamples (Sequence[bool]): Whether to downsample the feature map
            at each stage when combine is 'concat' in the first ShuffleUnit.
            Default: (True, True, True).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 groups=3,
                 widen_factor=1.0,
                 out_indices=(0, 1, 2),
                 stage_blocks=(4, 8, 4),
                 downsamples=(True, True, True),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None):
        super(ShuffleNetV1, self).__init__(init_cfg)
        assert len(stage_blocks) == len(downsamples)
        self.pretrained = pretrained
        self.init_cfg = init_cfg
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        act_cfg = copy.deepcopy(act_cfg)
        self.stage_blocks = stage_blocks
        self.groups = groups
        self.downsamples = downsamples

        for index in out_indices:
            if index not in range(len(stage_blocks)):
                raise ValueError(
                    'the item in out_indices must in '
                    f'range(0, {len(stage_blocks)}). But received {index}')

        if frozen_stages not in range(-1, len(stage_blocks)):
            raise ValueError(
                f'frozen_stages must be in range(-1, {len(stage_blocks)}). '
                f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        if groups == 1:
            channels = (144, 288, 576)
        elif groups == 2:
            channels = (200, 400, 800)
        elif groups == 3:
            channels = (240, 480, 960)
        elif groups == 4:
            channels = (272, 544, 1088)
        elif groups == 8:
            channels = (384, 768, 1536)
        else:
            raise ValueError(f'{groups} groups is not supported for 1x1 '
                             'Grouped Convolutions')

        channels = [make_divisible(ch * widen_factor, 8) for ch in channels]

        self.in_channels = int(24 * widen_factor)

        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList()
        for i, num_blocks in enumerate(self.stage_blocks):
            first_block = (i == 0)
            layer = self.make_layer(channels[i], num_blocks, first_block,
                                    downsamples[i])
            self.layers.append(layer)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(self.frozen_stages):
            layer = self.layers[i]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def make_layer(self,
                   out_channels,
                   num_blocks,
                   first_block=False,
                   downsample=True):
        """Stack ShuffleUnit blocks to make a layer.

        Args:
            out_channels (int): out_channels of the block.
            num_blocks (int): Number of blocks.
            first_block (bool, optional): Whether is the first ShuffleUnit of a
                sequential ShuffleUnits. Default: True, which means not using
                the grouped 1x1 convolution.
        """
        layers = []
        for i in range(num_blocks):
            first_block = first_block if i == 0 else False
            combine_mode = 'concat' if i == 0 else 'add'
            layers.append(
                ShuffleUnit(
                    self.in_channels,
                    out_channels,
                    groups=self.groups,
                    first_block=first_block,
                    combine=combine_mode,
                    downsample=downsample,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def init_weights(self):
        """Initialize the weights in backbone."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for name, m in self.named_modules():
                if isinstance(m, nn.Conv2d):
                    if 'conv1' in name:
                        normal_init(m, mean=0, std=0.01)
                    else:
                        normal_init(m, mean=0, std=1.0 / m.weight.shape[1])
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, val=1, bias=0.0001)
                    if isinstance(m, _BatchNorm):
                        if m.running_mean is not None:
                            nn.init.constant_(m.running_mean, 0)
        else:
            raise TypeError('pretrained must be a str or None. But received '
                            f'{type(self.pretrained)}')

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ShuffleNetV1, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
