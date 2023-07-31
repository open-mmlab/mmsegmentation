import torch
import torch.nn as nn
from mmseg.models.backbones.strideformer import ConvBNAct
from mmseg.registry import MODELS
import torch.nn.functional as F
from typing import List
from torch import Tensor


@MODELS.register_module()
class PPMobileSegHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 use_dw=True,
                 dropout_ratio=0.1,
                 align_corners=False,
                 upsample='intepolate',
                 out_channels=None):
        super().__init__()
        self.align_corners = align_corners
        self.last_channels = in_channels
        self.upsample = upsample
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.linear_fuse = ConvBNAct(
            in_channels=self.last_channels,
            out_channels=self.last_channels,
            kernel_size=1,
            stride=1,
            groups=self.last_channels if use_dw else 1,
            act=nn.ReLU)
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.conv_seg = nn.Conv2d(
            self.last_channels, self.num_classes, kernel_size=1)

    def forward(self, x):
        x, x_hw = x[0], x[1]
        x = self.linear_fuse(x)
        x = self.dropout(x)
        x = self.conv_seg(x)
        if self.upsample == 'intepolate' or self.training or self.num_classes < 30:
            x = F.interpolate(
                x, x_hw, mode='bilinear', align_corners=self.align_corners)
        elif self.upsample == 'vim':
            labelset = torch.unique(torch.argmax(x, 1))
            x = torch.gather(x, 1, labelset)
            x = F.interpolate(
                x, x_hw, mode='bilinear', align_corners=self.align_corners)

            pred = torch.argmax(x, 1)
            pred_retrieve = torch.zeros(pred.shape, dtype=torch.int32)
            for i, val in enumerate(labelset):
                pred_retrieve[pred == i] = labelset[i].cast('int32')

            x = pred_retrieve
        else:
            raise NotImplementedError(self.upsample, " is not implemented")

        return [x]

    def predict(self, inputs, batch_img_metas: List[dict], test_cfg,
                **kwargs) -> List[Tensor]:
        """Forward function for testing, only ``pam_cam`` is used."""
        seg_logits = self.forward(inputs)[0]
        return seg_logits
