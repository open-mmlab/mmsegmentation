import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from ..builder import BACKBONES
from ..utils import ResLayer
from .resnet import BasicBlock


class HourglassModule(BaseModule):
    """Hourglass Module for HourglassNet backbone.

    Generate module recursively and use BasicBlock as the base unit.

    Args:
        depth (int): Depth of current HourglassModule. This is also the
            downsampling times. len(stage_channels) should lagrer than
            downsample_times.
        stage_channels (list[int]): The number of feature channels of each
            stage in the current HourglassModule. len(stage_channels) should
            equal to len(stage_blocks).
        stage_blocks (list[int]): The number of stacked blocks of each stage in
            the current HourglassModule.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: `dict(type='BN', requires_grad=True)`.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: `None`.
        upsample_cfg (dict, optional): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`.
    """

    def __init__(self,
                 depth,
                 stage_channels,
                 stage_blocks,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 init_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(HourglassModule, self).__init__(init_cfg)

        self.depth = depth

        cur_block = stage_blocks[0]
        next_block = stage_blocks[1]

        cur_channel = stage_channels[0]
        next_channel = stage_channels[1]

        self.up1 = ResLayer(
            BasicBlock, cur_channel, cur_channel, cur_block, norm_cfg=norm_cfg)

        self.low1 = ResLayer(
            BasicBlock,
            cur_channel,
            next_channel,
            cur_block,
            stride=2,
            norm_cfg=norm_cfg)

        if self.depth > 1:
            self.low2 = HourglassModule(depth - 1, stage_channels[1:],
                                        stage_blocks[1:])
        else:
            self.low2 = ResLayer(
                BasicBlock,
                next_channel,
                next_channel,
                next_block,
                norm_cfg=norm_cfg)

        self.low3 = ResLayer(
            BasicBlock,
            next_channel,
            cur_channel,
            cur_block,
            norm_cfg=norm_cfg)

        self.up2 = F.interpolate
        self.upsample_cfg = upsample_cfg

    def forward(self, x):
        """Forward function."""
        up1 = self.up1(x)
        low1 = self.low1(x)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        # Fixing `scale factor` (e.g. 2) is common for upsampling, but
        # in some cases the spatial size is mismatched and error will arise.
        if 'scale_factor' in self.upsample_cfg:
            up2 = self.up2(low3, **self.upsample_cfg)
        else:
            shape = up1.shape[2:]
            up2 = self.up2(low3, size=shape, **self.upsample_cfg)
        return up1 + up2


@BACKBONES.register_module()
class HourglassNet(BaseModule):
    """HourglassNet backbone.

    Stacked Hourglass Networks for Human Pose Estimation.
    More details can be found in the `paper
    <https://arxiv.org/abs/1603.06937>`_ .

    Args:
        downsample_times (int): The number of downsample times in a
            HourglassModule. Default: `5`.
        num_stacks (int): The number of stacked HourglassModule in
            HourglassNet, 1 for Hourglass-52, 2 for Hourglass-104.
            Default: `2`.
        stage_channels (list[int]): The number of feature channels of each
            stage in a HourglassModule.
            Default: `(256, 256, 384, 384, 384, 512)`.
        stage_blocks (list[int]): The number of stacked blocks of each stage in
            a HourglassModule. Default: `(2, 2, 2, 2, 2, 4)`.
        feat_channel (int): The number of output feature channels each
            HourglassModule. Default: `256`.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: `dict(type='BN', requires_grad=True)`.
        pretrained (str, optional): model pretrained path. Default: `None`.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: `None`.

    Example:
        >>> from mmdet.models import HourglassNet
        >>> import torch
        >>> self = HourglassNet()
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 511, 511)
        >>> level_outputs = self.forward(inputs)
        >>> for level_output in level_outputs:
        ...     print(tuple(level_output.shape))
        (1, 256, 128, 128)
        (1, 256, 128, 128)
    """

    def __init__(self,
                 downsample_times=5,
                 num_stacks=2,
                 stage_channels=(256, 256, 384, 384, 384, 512),
                 stage_blocks=(2, 2, 2, 2, 2, 4),
                 feat_channel=256,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 pretrained=None,
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(HourglassNet, self).__init__(init_cfg)

        self.num_stacks = num_stacks
        assert self.num_stacks >= 1
        assert len(stage_channels) == len(stage_blocks)
        assert len(stage_channels) > downsample_times

        cur_channel = stage_channels[0]

        self.stem = nn.Sequential(
            ConvModule(3, 128, 7, padding=3, stride=2, norm_cfg=norm_cfg),
            ResLayer(
                BasicBlock, 128, cur_channel, 1, stride=2, norm_cfg=norm_cfg))

        self.hourglass_modules = nn.ModuleList([
            HourglassModule(downsample_times, stage_channels, stage_blocks)
            for _ in range(num_stacks)
        ])

        self.inters = ResLayer(
            BasicBlock,
            cur_channel,
            cur_channel,
            num_stacks - 1,
            norm_cfg=norm_cfg)

        self.conv1x1s = nn.ModuleList([
            ConvModule(
                cur_channel, cur_channel, 1, norm_cfg=norm_cfg, act_cfg=None)
            for _ in range(num_stacks - 1)
        ])

        self.out_convs = nn.ModuleList([
            ConvModule(
                cur_channel, feat_channel, 3, padding=1, norm_cfg=norm_cfg)
            for _ in range(num_stacks)
        ])

        self.remap_convs = nn.ModuleList([
            ConvModule(
                feat_channel, cur_channel, 1, norm_cfg=norm_cfg, act_cfg=None)
            for _ in range(num_stacks - 1)
        ])

        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        """Init module weights."""
        # Training Centripetal Model needs to reset parameters for Conv2d
        super(HourglassNet, self).init_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()

    def forward(self, x):
        """Forward function."""
        inter_feat = self.stem(x)
        out_feats = []

        for idx in range(self.num_stacks):
            hourglass_feat = self.hourglass_modules[idx](inter_feat)
            out_feat = self.out_convs[idx](hourglass_feat)
            out_feats.append(out_feat)

            if idx < self.num_stacks - 1:
                inter_feat = self.conv1x1s[idx](
                    inter_feat) + self.remap_convs[idx](
                        out_feat)
                inter_feat = self.inters[idx](self.relu(inter_feat))

        return out_feats
