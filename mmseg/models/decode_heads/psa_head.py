import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.ops import ConvModule, PSAMask
from ..registry import HEADS
from .decode_head import DecodeHead


@HEADS.register_module
class PSAHead(DecodeHead):

    def __init__(self,
                 psa_type='bi-direction',
                 compact=False,
                 shrink_factor=2,
                 mask_size=(89, 89),
                 normalization_factor=1.0,
                 psa_softmax=True,
                 **kwargs):
        super(PSAHead, self).__init__(**kwargs)
        assert psa_type in ['collect', 'distribute', 'bi-direction']
        self.psa_type = psa_type
        self.compact = compact
        self.shrink_factor = shrink_factor
        self.mask_size = mask_size
        mask_h, mask_w = mask_size
        self.psa_softmax = psa_softmax
        if normalization_factor is None:
            normalization_factor = mask_h * mask_w
        self.normalization_factor = normalization_factor

        self.reduce = ConvModule(
            self.in_channels,
            self.channels,
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.attention = nn.Sequential(
            ConvModule(
                self.channels,
                self.channels,
                kernel_size=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            nn.Conv2d(
                self.channels, mask_h * mask_w, kernel_size=1, bias=False))
        if psa_type == 'bi-direction':
            self.reduce_p = ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            self.attention_p = nn.Sequential(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                nn.Conv2d(
                    self.channels, mask_h * mask_w, kernel_size=1, bias=False))
            self.psamask_collect = PSAMask('collect', mask_size)
            self.psamask_distribute = PSAMask('distribute', mask_size)
        else:
            self.psamask = PSAMask(psa_type, mask_size)
        self.proj = ConvModule(
            self.channels * (2 if psa_type == 'bi-direction' else 1),
            self.in_channels,
            kernel_size=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck = ConvModule(
            self.in_channels * 2,
            self.channels,
            kernel_size=3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        identity = x
        if self.psa_type in ['collect', 'distribute']:
            out = self.reduce(x)
            n, c, h, w = out.size()
            if self.shrink_factor != 1:
                h = (h - 1) // self.shrink_factor + 1
                w = (w - 1) // self.shrink_factor + 1
                out = F.interpolate(
                    out, size=(h, w), mode='bilinear', align_corners=True)
            y = self.attention(out)
            if self.compact:
                if self.psa_type == 'collect':
                    y = y.view(n, h * w,
                               h * w).transpose(1, 2).view(n, h * w, h, w)
            else:
                y = self.psamask(y)
            if self.psa_softmax:
                y = F.softmax(y, dim=1)
            out = torch.bmm(
                out.view(n, c, h * w), y.view(n, h * w, h * w)).view(
                    n, c, h, w) * (1.0 / self.normalization_factor)
        else:
            x_col = self.reduce(x)
            x_dis = self.reduce_p(x)
            n, c, h, w = x_col.size()
            if self.shrink_factor != 1:
                h = (h - 1) // self.shrink_factor + 1
                w = (w - 1) // self.shrink_factor + 1
                x_col = F.interpolate(
                    x_col, size=(h, w), mode='bilinear', align_corners=True)
                x_dis = F.interpolate(
                    x_dis, size=(h, w), mode='bilinear', align_corners=True)
            y_col = self.attention(x_col)
            y_dis = self.attention_p(x_dis)
            if self.compact:
                y_dis = y_dis.view(n, h * w,
                                   h * w).transpose(1, 2).view(n, h * w, h, w)
            else:
                y_col = self.psamask_collect(y_col)
                y_dis = self.psamask_distribute(y_dis)
            if self.psa_softmax:
                y_col = F.softmax(y_col, dim=1)
                y_dis = F.softmax(y_dis, dim=1)
            x_col = torch.bmm(
                x_col.view(n, c, h * w), y_col.view(n, h * w, h * w)).view(
                    n, c, h, w) * (1.0 / self.normalization_factor)
            x_dis = torch.bmm(
                x_dis.view(n, c, h * w), y_dis.view(n, h * w, h * w)).view(
                    n, c, h, w) * (1.0 / self.normalization_factor)
            out = torch.cat([x_col, x_dis], 1)
        out = self.proj(out)
        if self.shrink_factor != 1:
            h = (h - 1) * self.shrink_factor + 1
            w = (w - 1) * self.shrink_factor + 1
            out = F.interpolate(
                out, size=(h, w), mode='bilinear', align_corners=True)
        out = self.bottleneck(torch.cat((identity, out), dim=1))
        out = self.cls_seg(out)
        return out
