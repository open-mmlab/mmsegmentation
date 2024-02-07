# -----------------------------------------------------------------------------------
#  Modified from
# https://github.com/OpenGVLab/InternImage 
# ------------------------------------------------------------------------------------

import math
import warnings
import torch
import torch.nn as nn
import sys
sys.path.append("./ops_dcnv3")
import modules as dcnv3
from mmengine.model import BaseModule

#TODO we might want to modify the dcnv3 to remove some layers(example the dw conv) to better fit the KA module
# husk att de gjør kernel decomposition inne i dcvn3, dette burde vi endre på i strip wise siden det løser samme problemet
class DCNv3KA(BaseModule):
    def __init__(self,
                 core_op,
                 channels,
                 groups,
                 kernel_size,
                 stride,
                 pad,
                 dilation,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 layer_scale=None,
                 offset_scale=1.0,
                 dw_kernel_size=None, # for InternImage-H/G
                 res_post_norm=False, # for InternImage-H/G
                 center_feature_scale=False): 
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.core_op = getattr(dcnv3, core_op)
        self.dcn = self.core_op(
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            pad=pad,
            dilation=dilation,
            group=16,
            offset_scale=offset_scale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            dw_kernel_size=dw_kernel_size, # for InternImage-H/G
            center_feature_scale=center_feature_scale)
        self.ca = nn.Conv2d(self.channels, self.channels, 1)

    def forward(self, x):
        u = x.clone()
        u = u.permute(0,3,1,2)
        x = self.dcn(x)
        x = x.permute(0,3,1,2)
        attn = self.ca(x)
        return u * attn

# TODO lag også en egen som likner mer på intern Image