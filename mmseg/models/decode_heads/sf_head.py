import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from .decode_head import BaseDecodeHead
from ..builder import HEADS


class AlignModule(nn.ModuleList):
    def __init__(self, inplane, outplane):
        super(AlignModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, kernel_size=1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, kernel_size=1, bias=False)
        self.flow_make = nn.Conv2d(outplane * 2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        l_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = l_feature.size()[2:]
        size = (h, w)
        l_feature = self.down_l(l_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.upsample(h_feature, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, l_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)
        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(input, grid)
        return output

class PSPModule(nn.Module):
    def __init__(self, in_channels, out_features=512, pool_scales=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, out_features, scale, norm_layer) for scale in pool_scales])
        self.bottleneck = nn.Sequential(
            ConvModule(
                in_channels + len(pool_scales) * out_features,
                out_features,
                kernel_size=1,
                norm_cfg=dict(type='BN', requires_grad=True)),
            nn.Dropout2d(0.1))

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

@HEADS.register_module()
class SFHead(BaseDecodeHead):
    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(SFHead, self).__init__(**kwargs)
        assert isinstance(pool_scales, (list, tuple))
        self.pool_scales = pool_scales
        self.ppm = PSPModule(
            in_channels=self.in_channels,
            out_features=self.channels)
        fpn_inplanes = [self.in_channels // 8, self.in_channels // 4, self.in_channels // 2, self.in_channels]
        self.fpn_in = nn.ModuleList()
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                    ConvModule(
                        fpn_inplane,
                        self.channels,
                        kernel_size=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
        self.fpn_out = nn.ModuleList()
        self.fpn_out_align = nn.ModuleList()
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(
                    ConvModule(
                        self.channels,
                        self.channels,
                        kernel_size=3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            self.fpn_out_align.append(
                AlignModule(inplane=self.channels, outplane=self.channels // 2)
            )
        self.conv_last = nn.Sequential(
            ConvModule(
                4 * self.channels,
                self.channels,
                kernel_size=3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                self.channels,
                self.num_classes,
                kernel_size=1)
        )

    def forward(self, x):
        psp_out = self.ppm(x[-1])
        f = psp_out
        fpn_feature_list = [psp_out]
        for i in reversed(range(len(x) - 1)):
            conv_x = x[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch
            f = self.fpn_out_align[i]([conv_x, f])
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(F.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear',
                align_corners=True))
        output = torch.cat(fusion_list, 1)
        output = self.cls_seg(output)
        return output
