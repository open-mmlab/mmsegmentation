import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm

from mmseg.models.backbones.mobile_net_v2 import InvertedResidual
from mmseg.models.decode_heads.psp_head import PPM
from mmseg.ops import DepthwiseSeparableConvModule, resize
from ..builder import BACKBONES


class LearningToDownsample(nn.Module):
    """Learning to downsample module.
    Args:
        in_channels (int): Number of input channels.

        dw_channels1 (int): Number of output channels of the first
            depthwise conv (dwconv) layer.

        dw_channels2 (int): Number of output channels of the second
            dwconv layer.

        out_channels (int): Number of output channels of the whole
            'learning to downsample' module.


        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self,
                 in_channels,
                 dw_channels1,
                 dw_channels2,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(LearningToDownsample, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.conv = ConvModule(
            in_channels,
            dw_channels1,
            3,
            stride=2,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.dsconv1 = DepthwiseSeparableConvModule(
            dw_channels1,
            dw_channels2,
            stride=2,
            relu_first=False,
            norm_cfg=self.norm_cfg)
        self.dsconv2 = DepthwiseSeparableConvModule(
            dw_channels2,
            out_channels,
            stride=2,
            relu_first=False,
            norm_cfg=self.norm_cfg)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module.
    Args:
        in_channels (int): Number of input channels of the GFE module.

        block_channels (tuple): Tuple of ints. Each int specifies the
            number of output channels of each Inverted Residual module.

        out_channels(int): Number of output channels of the GFE module.

        t (int): t parameter (upsampling factor) of each Inverted Residual
            module.

        num_blocks (tuple): Tuple of ints. Each int specifies the number of
        times each Inverted Residual module is repeated.

        pool_scales (tuple): Tuple of ints. Each int specifies the parameter
            required in 'global average pooling' within PPM.

        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self,
                 in_channels=64,
                 block_channels=(64, 96, 128),
                 out_channels=128,
                 t=6,
                 num_blocks=(3, 3, 3),
                 pool_scales=(1, 2, 3, 6),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 align_corners=True):
        super(GlobalFeatureExtractor, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        assert len(block_channels) == len(num_blocks) == 3
        self.bottleneck1 = self._make_layer(in_channels, block_channels[0],
                                            num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(block_channels[0],
                                            block_channels[1], num_blocks[1],
                                            t, 2)
        self.bottleneck3 = self._make_layer(block_channels[1],
                                            block_channels[2], num_blocks[2],
                                            t, 1)
        self.ppm = PPM(
            pool_scales,
            block_channels[2],
            block_channels[2] // 4,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=align_corners)
        self.out = ConvModule(
            block_channels[2] * 2,
            out_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def _make_layer(self, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(
            InvertedResidual(
                inplanes, planes, stride, t, norm_cfg=self.norm_cfg))
        for i in range(1, blocks):
            layers.append(
                InvertedResidual(planes, planes, 1, t, norm_cfg=self.norm_cfg))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = torch.cat([x, *self.ppm(x)], dim=1)
        x = self.out(x)
        return x


class FeatureFusionModule(nn.Module):
    """Feature fusion module.
    Args:
        higher_in_channels (int): Number of input channels of the
            higher-resolution branch.

        lower_in_channels (int): Number of input channels of the
            lower-resolution branch.

        out_channels (int): Number of output channels.

        scale_factor (int): Scale factor applied to the lower-res input.
            Should be coherent with the downsampling factor determined
            by the GFE module.

        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self,
                 higher_in_channels,
                 lower_in_channels,
                 out_channels,
                 scale_factor,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 align_corners=True):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.dwconv = ConvModule(
            lower_in_channels,
            out_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.conv_lower_res = ConvModule(
            out_channels,
            out_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.conv_higher_res = ConvModule(
            higher_in_channels,
            out_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = resize(
            lower_res_feature,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=self.align_corners)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


@BACKBONES.register_module()
class FastSCNN(nn.Module):
    """Fast-SCNN Backbone.
    Args:
        in_channels (int): Number of input image channels. Default=3 (RGB)

        downsample_dw_channels1 (int): Number of output channels after
            the first conv layer in Learning-To-Downsample (LTD) module.

        downsample_dw_channels2 (int): Number of output channels
            after the second conv layer in LTD.

        global_in_channels (int): Number of input channels of
            Global Feature Extractor(GFE).
            Equal to number of output channels of LTD.

        global_block_channels (tuple): Tuple of integers that describe
            the output channels for each of the MobileNet-v2 bottleneck
            residual blocks in GFE.

        global_out_channels (int): Number of output channels of GFE.

        higher_in_channels (int): Number of input channels of the higher
            resolution branch in FFM.
            Equal to global_in_channels.

        lower_in_channels (int): Number of input channels of  the lower
            resolution branch in FFM.
            Equal to global_out_channels.

        fusion_out_channels (int): Number of output channels of FFM.

        scale_factor (int): The upsampling factor of the higher resolution
            branch in FFM.
            Equal to the downsampling factor in GFE.

        out_indices (tuple): Tuple of indices of list
            [higher_res_features, lower_res_features, fusion_output].
            Often set to (0,1,2) to enable aux. heads.

        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self,
                 in_channels=3,
                 downsample_dw_channels1=32,
                 downsample_dw_channels2=48,
                 global_in_channels=64,
                 global_block_channels=(64, 96, 128),
                 global_out_channels=128,
                 higher_in_channels=64,
                 lower_in_channels=128,
                 fusion_out_channels=128,
                 scale_factor=4,
                 out_indices=(0, 1, 2),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False):

        super(FastSCNN, self).__init__()
        if global_in_channels != higher_in_channels:
            raise AssertionError('Global Input Channels must be the same \
                                 with Higher Input Channels!')
        elif global_out_channels != lower_in_channels:
            raise AssertionError('Global Output Channels must be the same \
                                with Lower Input Channels!')
        if scale_factor != 4:
            raise AssertionError('Scale-factor must compensate the \
                            downsampling factor in the GFE module!')

        self.in_channels = in_channels
        self.downsample_dw_channels1 = downsample_dw_channels1
        self.downsample_dw_channels2 = downsample_dw_channels2
        self.global_in_channels = global_in_channels
        self.global_block_channels = global_block_channels
        self.global_out_channels = global_out_channels
        self.higher_in_channels = higher_in_channels
        self.lower_in_channels = lower_in_channels
        self.fusion_out_channels = fusion_out_channels
        self.scale_factor = scale_factor
        self.out_indices = out_indices
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.learning_to_downsample = LearningToDownsample(
            in_channels,
            downsample_dw_channels1,
            downsample_dw_channels2,
            global_in_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.global_feature_extractor = GlobalFeatureExtractor(
            global_in_channels,
            global_block_channels,
            global_out_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.feature_fusion = FeatureFusionModule(
            higher_in_channels,
            lower_in_channels,
            fusion_out_channels,
            scale_factor=self.scale_factor,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x):
        higher_res_features = self.learning_to_downsample(x)
        lower_res_features = self.global_feature_extractor(higher_res_features)
        fusion_output = self.feature_fusion(higher_res_features,
                                            lower_res_features)

        outs = [higher_res_features, lower_res_features, fusion_output]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)
