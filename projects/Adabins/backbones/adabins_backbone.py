import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmengine.model import BaseModule

from mmseg.registry import MODELS


class UpSampleBN(nn.Module):
    """ UpSample module
    Args:
        skip_input (int): the input feature
        output_features (int): the output feature
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict, optional): The activation layer of AAM:
            Aggregate Attention Module.
    """

    def __init__(self,
                 skip_input,
                 output_features,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='LeakyReLU')):
        super().__init__()

        self._net = nn.Sequential(
            ConvModule(
                in_channels=skip_input,
                out_channels=output_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            ConvModule(
                in_channels=output_features,
                out_channels=output_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ))

    def forward(self, x, concat_with):
        up_x = F.interpolate(
            x,
            size=[concat_with.size(2),
                  concat_with.size(3)],
            mode='bilinear',
            align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class Encoder(nn.Module):
    """ the efficientnet_b5 model
    Args:
        basemodel_name (str): the name of base model
    """

    def __init__(self, basemodel_name):
        super().__init__()
        self.original_model = timm.create_model(
            basemodel_name, pretrained=True)
        # Remove last layer
        self.original_model.global_pool = nn.Identity()
        self.original_model.classifier = nn.Identity()

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if k == 'blocks':
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


@MODELS.register_module()
class AdabinsBackbone(BaseModule):
    """ the backbone of the adabins
    Args:
        basemodel_name (str):the name of base model
        num_features (int): the middle feature
        num_classes (int): the classes number
        bottleneck_features (int): the bottleneck features
        conv_cfg (dict): Config dict for convolution layer.
    """

    def __init__(self,
                 basemodel_name,
                 num_features=2048,
                 num_classes=128,
                 bottleneck_features=2048,
                 conv_cfg=dict(type='Conv')):
        super().__init__()
        self.encoder = Encoder(basemodel_name)
        features = int(num_features)
        self.conv2 = build_conv_layer(
            conv_cfg,
            bottleneck_features,
            features,
            kernel_size=1,
            stride=1,
            padding=1)
        self.up1 = UpSampleBN(
            skip_input=features // 1 + 112 + 64, output_features=features // 2)
        self.up2 = UpSampleBN(
            skip_input=features // 2 + 40 + 24, output_features=features // 4)
        self.up3 = UpSampleBN(
            skip_input=features // 4 + 24 + 16, output_features=features // 8)
        self.up4 = UpSampleBN(
            skip_input=features // 8 + 16 + 8, output_features=features // 16)

        self.conv3 = build_conv_layer(
            conv_cfg,
            features // 16,
            num_classes,
            kernel_size=3,
            stride=1,
            padding=1)

    def forward(self, x):
        features = self.encoder(x)
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[
            3], features[4], features[5], features[7], features[10]
        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        out = self.conv3(x_d4)
        return out
