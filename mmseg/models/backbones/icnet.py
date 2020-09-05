import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init, normal_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmseg.ops import resize
from mmseg.utils import get_root_logger
from ..builder import BACKBONES
from ..decode_heads.psp_head import PSPHead
from .resnet import ResNet


@BACKBONES.register_module()
class ICNet(nn.Module):
    """ICNet backbone

    ICNet for Real-Time Semantic Segmentation on High-Resolution Images
    arXiv: https://arxiv.org/abs/1704.08545

    Args:
        in_channels (int): Number of input image channels. Normally 3.
        num_channels (list[int]): Numbers of feature channels at each branches.
        resnet_cfg (dict): Config dict to build ResNet.
        conv_cfg (dict): Dictionary to construct and config conv layer.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        act_cfg (dict): Dictionary to construct and config act layer.
    """

    def __init__(self,
                 in_channels=3,
                 num_channels=[64, 256, 256],
                 resnet_cfg=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):

        super(ICNet, self).__init__()
        self.backbone = ResNet(**resnet_cfg)
        self.backbone.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=True)

        self.psp_head = PSPHead(
            in_channels=2048,
            channels=512,
            num_classes=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        del self.psp_head.conv_seg

        self.conv_sub1 = nn.Sequential(
            ConvModule(
                in_channels, 32, 3, 2, 1, conv_cfg=conv_cfg,
                norm_cfg=norm_cfg),
            ConvModule(32, 32, 3, 2, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg),
            ConvModule(
                32,
                num_channels[0],
                3,
                2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg))

        self.conv_sub2 = ConvModule(
            512, num_channels[1], 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg)

        self.conv_sub4 = ConvModule(
            512, num_channels[2], 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg)

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, mode='fan_out')
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, 0, 0.01)

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(
                self.backbone, pretrained, strict=False, logger=logger)

    def forward(self, x):
        output = []

        # sub 1
        output.append(self.conv_sub1(x))

        # sub 2
        x = resize(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x = self.backbone.stem(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        output.append(self.conv_sub2(x))

        # sub 4
        x = resize(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        psp_outs = self.psp_head.psp_modules(x) + [x]
        psp_outs = torch.cat(psp_outs, dim=1)
        x = self.psp_head.bottleneck(psp_outs)

        output.append(self.conv_sub4(x))

        return output
