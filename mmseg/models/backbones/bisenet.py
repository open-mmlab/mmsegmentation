import warnings

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init, trunc_normal_init
from mmcv.runner import BaseModule, _load_checkpoint

from mmseg.utils import get_root_logger
from ..builder import BACKBONES


class DetailBranch(BaseModule):
    """Detail Branch with wide channels and shallow layers to capture low-level
    details and generate high-resolution feature representation.

    Args:
        db_channels (Tuple[int]): Size of channel numbers of Stage 1, Stage 2
            and Stage 3 in Detail Branch.
            Default: (64, 64, 128).
        conv_cfg (dict | None): Config of conv layers.
            Default: None
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN')
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')

    Returns:
        feat (Tensor): Feature Map of Detail Branch.
    """

    def __init__(self,
                 db_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(DetailBranch, self).__init__()

        C1, C2, C3 = db_channels

        self.S1 = nn.Sequential(
            ConvModule(
                3,
                C1,
                3,
                stride=2,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                C1,
                C1,
                3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
        )
        self.S2 = nn.Sequential(
            ConvModule(
                C1,
                C2,
                3,
                stride=2,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                C2,
                C2,
                3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                C2,
                C2,
                3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
        )
        self.S3 = nn.Sequential(
            ConvModule(
                C2,
                C3,
                3,
                stride=2,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                C3,
                C3,
                3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                C3,
                C3,
                3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
        )

    def forward(self, x):
        feat_d = self.S1(x)
        feat_d = self.S2(feat_d)
        feat_d = self.S3(feat_d)
        return feat_d


class StemBlock(BaseModule):
    """Stem Block which uses two different downsampling manners to shrink the
    feature representation.

        As illustrated in Fig. 4 (a), the left branch is two ConvModules and
        the right is MaxPooling. The output feature of both
        branches are concatenated as the output.

    Args:
        in_channels (int): Number of input channels.
            Default: 3.
        out_channels (int): Number of output channels.
            Default: 16.
        conv_cfg (dict | None): Config of conv layers.
            Default: None
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN')
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')

    Returns:
        feat (Tensor): Feature Map `feat2` in Semantic Branch.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(StemBlock, self).__init__()

        self.conv = ConvModule(
            in_channels,
            out_channels,
            3,
            stride=2,
            padding=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.left = nn.Sequential(
            ConvModule(
                out_channels,
                out_channels // 2,
                1,
                stride=1,
                padding=0,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                out_channels // 2,
                out_channels,
                3,
                stride=2,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvModule(
            out_channels * 2,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat_cat = torch.cat([feat_left, feat_right], dim=1)
        feat2 = self.fuse(feat_cat)
        return feat2


class GELayerS1(BaseModule):
    """Gather-and-Expansion Layer with Stride 1.

        As illustrated in Fig. 5 (b), it is a bottleneck structure.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        conv_cfg (dict | None): Config of conv layers.
            Default: None
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN')
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')

    Returns:
        feat (Tensor): Intermidiate Feature Map for
            next layer in Semantic Branch.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 exp_ratio=6,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(GELayerS1, self).__init__()
        mid_channel = in_channels * exp_ratio
        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_channels,
                bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_channel,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat


class GELayerS2(BaseModule):
    """Gather-and-Expansion Layer with Stride 2.

        As illustrated in Fig. 5 (c), when stride=2, two 3x3 depth-wise
        convolution and one 3x3 separable convolution are adopted
        as shortcut in the bottleneck structure.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        exp_ratio (int): Expansion ratio for middle channels.
            Default: 6.
        conv_cfg (dict | None): Config of conv layers.
            Default: None
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN')
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')

    Returns:
        feat (Tensor): Intermidiate Feature Map for next
            layer in Semantic Branch.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 exp_ratio=6,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(GELayerS2, self).__init__()

        mid_channel = in_channels * exp_ratio
        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_channels,
                bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(
                mid_channel,
                mid_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=mid_channel,
                bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_channel,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_channels,
                bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat


class CEBlock(BaseModule):
    """Context Embedding Block for large receptive filed in Semantic Branch. As
    illustrated in Fig. 4 (b), it is designed with the global average pooling
    to embed the global contextual information.

    Args:
        in_channels (int): Number of input channels.
            Default: 3.
        out_channels (int): Number of output channels.
            Default: 16.
        conv_cfg (dict | None): Config of conv layers.
            Default: None
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN')
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')

    Returns:
        feat (Tensor): Feature Map `feat5_5` in Semantic Branch.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(CEBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Not changed to SyncBatchNorm Yet.
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.conv_gap = ConvModule(
            self.in_channels,
            self.out_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        # TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvModule(
            self.out_channels,
            self.out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat5_5 = self.conv_last(feat)
        return feat5_5


class SegmentBranch(BaseModule):
    """Semantic Branch which is lightweight with narrow channels and deep
    layers to obtain　high-level semantic context. As illustrated in Fig. 4 (b),
    it is designed with　the global average pooling to embed the global
    contextual information.

    Args:
        sb_channels (int): (Tuple[int]): Size of channel numbers of
            Stage 1, Stage 3, Stage 4 and Stage 5 in Semantic Branch.
            Default: (16, 32, 64, 128).
        exp_ratio (int): Expansion ratio for middle channels.
            Default: 6.

    Returns:
        feats (Tensor): Several Feature Maps `feat2`, `feat3`, `feat4`,
        `feat5_4` and `feat5_5` for auxiliary heads (Booster) and
        Bilateral Guided Aggregation Layer.
    """

    def __init__(self, sb_channels, exp_ratio=6):
        super(SegmentBranch, self).__init__()

        C1, C3, C4, C5 = sb_channels

        self.S1S2 = StemBlock(3, C1)
        self.S3 = nn.Sequential(
            # Gather And Expansion Layer With Stride 2 and 1, respectively.
            GELayerS2(C1, C3, exp_ratio),
            GELayerS1(C3, C3, exp_ratio),
        )
        self.S4 = nn.Sequential(
            GELayerS2(C3, C4, exp_ratio),
            GELayerS1(C4, C4, exp_ratio),
        )
        self.S5_4 = nn.Sequential(
            GELayerS2(C4, C5, exp_ratio),
            GELayerS1(C5, C5, exp_ratio),
            GELayerS1(C5, C5, exp_ratio),
            GELayerS1(C5, C5, exp_ratio),
        )
        self.S5_5 = CEBlock(C5, C5)

    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5


class BGALayer(BaseModule):
    """Bilateral Guided Aggregation Layer to fuse the complementary information
    from both Detail Branch and Semantic Branch. As illustrated in Fig. 3 & 6,
    this layer employs the contextual information of Semantic Branch to guide
    the feature response of Detail Branch.

    Args:
        out_channels (int): Number of output channels.
            Default: 128.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        conv_cfg (dict | None): Config of conv layers.
            Default: None
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN')
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')

    Returns:
        feat (Tensor): Output feature map for Segment heads.
    """

    def __init__(self,
                 out_channels,
                 align_corners,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(BGALayer, self).__init__()

        self.out_channels = out_channels
        self.align_corners = align_corners
        self.left1 = nn.Sequential(
            nn.Conv2d(
                self.out_channels,
                self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=self.out_channels,
                bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.Conv2d(
                self.out_channels,
                self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                self.out_channels,
                self.out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False), nn.BatchNorm2d(self.out_channels),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        self.right1 = nn.Sequential(
            nn.Conv2d(
                self.out_channels,
                self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                self.out_channels,
                self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=self.out_channels,
                bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.Conv2d(
                self.out_channels,
                self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
        )

        # TODO: does this really has no relu?
        self.conv = ConvModule(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            inplace=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x_d, x_s):
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = nn.functional.interpolate(input=right1, scale_factor=4)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = nn.functional.interpolate(input=right, scale_factor=4)
        output = self.conv(left + right)
        return output


@BACKBONES.register_module()
class BiSeNetV2(BaseModule):
    """BiSeNetV2: Bilateral Network with Guided Aggregation for
        Real-time Semantic Segmentation

        This backbone is the implementation of
        `BiSeNetV2 <https://arxiv.org/abs/2004.02147>`_.

    Args:
        pretrained (str, optional): The model pretrained path. Default: None.
        out_indices (Sequence[int] | int, optional): Output from which stages.
            Default: (0, 1, 2, 3).
        detail_branch_channels (Sequence[int], optional): Channels of S1, S2
            and S3 in Detail Branch, respectively. Default: (64, 64, 128).
        channel_ratio (float, optional): The ratio factor controls the size of
            channels in Semantic Branch. Default: 0.25.
        expansion_ratio (int, optional): The expansion factor expanding channel
            number in Semantic Branch. Default: 6
        align_corners (bool, optional): The align_corners argument of
            F.interpolate. Default: False.
        middle_channels (int, optional): Number of middle channels in
            Bilateral Guided Aggregation Layer. Default: 128.
        conv_cfg (dict | None): Config of conv layers.
            Default: None.
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 pretrained=None,
                 out_indices=(0, 1, 2, 3, 4),
                 detail_branch_channels=(64, 64, 128),
                 channel_ratio=0.25,
                 expansion_ratio=6,
                 align_corners=False,
                 middle_channels=128,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None,
                 **kwargs):
        super(BiSeNetV2, self).__init__(**kwargs)

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='Conv2d'),
                dict(
                    type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
            ]

        if isinstance(pretrained, str) or pretrained is None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
        else:
            raise TypeError('pretrained must be a str or None')

        self.pretrained = pretrained
        self.out_indices = out_indices
        self.detail_branch_channels = detail_branch_channels
        self.channel_ratio = channel_ratio
        self.expansion_ratio = expansion_ratio
        self.align_corners = align_corners
        self.middle_channels = middle_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        C1_d, C2_d, C3_d = self.detail_branch_channels
        db_channels = (C1_d, C2_d, C3_d)
        C1_s, C3_s, C4_s, C5_s = int(C1_d * channel_ratio), int(
            C3_d * channel_ratio), 64, 128
        sb_channels = (C1_s, C3_s, C4_s, C5_s)

        self.detail = DetailBranch(db_channels)
        self.segment = SegmentBranch(sb_channels, self.expansion_ratio)
        self.bga = BGALayer(self.middle_channels, self.align_corners)

    def init_weights(self):
        if self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m.weight, std=.02)
                    if m.bias is not None:
                        constant_init(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m.bias, 0)
                    constant_init(m.weight, 1.0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m.weight, mode='fan_in')
                    if m.bias is not None:
                        constant_init(m.bias, 0)
        elif isinstance(self.pretrained, str):
            logger = get_root_logger()
            checkpoint = _load_checkpoint(
                self.pretrained, logger=logger, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            self.load_state_dict(state_dict, False)

    def forward(self, x):
        feat_d = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        feat_head = self.bga(feat_d, feat_s)

        outs = [feat_head, feat2, feat3, feat4, feat5_4]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)
