import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from mmseg.ops import resize
from ..builder import BACKBONES, build_backbone
from ..decode_heads.psp_head import PPM


@BACKBONES.register_module()
class ICNet(BaseModule):
    """ICNet backbone.

    ICNet for Real-Time Semantic Segmentation on High-Resolution Images
    arXiv: https://arxiv.org/abs/1704.08545

    Args:
        in_channels (int): The number of input image channels. Default: 3.
        layer_channels (Sequence[int]): The numbers of feature channels at
            layer 2 and layer 4 in ResNet. It can also be other backbones.
            Default: (512, 2048).
        light_branch_middle_channels (int): The number of channels of the
            middle layer in light branch. Default: 32.
        psp_out_channels (int): The number of channels of the output of PSP
            module. Default: 512.
        out_channels (Sequence[int]): The numbers of output feature channels
            at each branches. Default: (64, 256, 256).
        resnet_cfg (dict): Config dict to build ResNet. It can also be other
            backbones. Default: None.
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
        conv_cfg (dict): Dictionary to construct and config conv layer.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        act_cfg (dict): Dictionary to construct and config act layer.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels=3,
                 layer_channels=(512, 2048),
                 light_branch_middle_channels=32,
                 psp_out_channels=512,
                 out_channels=(64, 256, 256),
                 resnet_cfg=None,
                 pool_scales=(1, 2, 3, 6),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False,
                 init_cfg=[
                     dict(type='Kaiming', mode='fan_out', layer='Conv2d'),
                     dict(type='Constant', val=1, layer='_BatchNorm'),
                     dict(type='Normal', mean=0.01, layer='Linear')
                 ]):

        super(ICNet, self).__init__(init_cfg=init_cfg)
        self.align_corners = align_corners
        self.backbone = build_backbone(resnet_cfg)
        self.backbone.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=True)

        # self.psp_head = PSPHead(
        #     in_channels=layer_channels[1],
        #     channels=psp_out_channels,
        #     num_classes=1,
        #     conv_cfg=conv_cfg,
        #     norm_cfg=norm_cfg,
        #     act_cfg=act_cfg)
        # del self.psp_head.conv_seg

        # RuntimeError: module did not have attribute conv_seg, but init_cfg
        # is {'type': 'Normal', 'std': 0.01}.
        # To avoid above error, we define psp_module and bottleneck seperatly.

        self.psp_modules = PPM(
            pool_scales=pool_scales,
            in_channels=layer_channels[1],
            channels=psp_out_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            align_corners=align_corners)

        self.psp_bottleneck = ConvModule(
            layer_channels[1] + len(pool_scales) * psp_out_channels,
            psp_out_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv_sub1 = nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=light_branch_middle_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg),
            ConvModule(
                in_channels=light_branch_middle_channels,
                out_channels=light_branch_middle_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg),
            ConvModule(
                in_channels=light_branch_middle_channels,
                out_channels=out_channels[0],
                kernel_size=3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg))

        self.conv_sub2 = ConvModule(
            layer_channels[0],
            out_channels[1],
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

        self.conv_sub4 = ConvModule(
            psp_out_channels,
            out_channels[2],
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

    def forward(self, x):
        output = []

        # sub 1
        output.append(self.conv_sub1(x))

        # sub 2
        x = resize(
            x,
            scale_factor=0.5,
            mode='bilinear',
            align_corners=self.align_corners)
        x = self.backbone.stem(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        output.append(self.conv_sub2(x))

        # sub 4
        x = resize(
            x,
            scale_factor=0.5,
            mode='bilinear',
            align_corners=self.align_corners)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        psp_outs = self.psp_modules(x) + [x]
        psp_outs = torch.cat(psp_outs, dim=1)
        x = self.psp_bottleneck(psp_outs)

        output.append(self.conv_sub4(x))

        return output
