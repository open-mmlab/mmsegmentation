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

# husk att de gjør kernel decomposition inne i dcvn3, dette burde vi endre på i strip wise siden det løser samme problemet
class DCNv3KA(BaseModule):
    def __init__(self,
                 core_op,
                 channels,
                 group,
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
        self.core_op = getattr(dcnv3, 'DCNv3_pytorch')
        self.output_proj = nn.Linear(channels, channels)
        self.dw_dcn = self.core_op(
            channels=channels,
            kernel_size=5,
            stride=1,
            pad=2,
            dilation=1,
            group=channels,
            offset_scale=offset_scale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            dw_kernel_size=dw_kernel_size, # for InternImage-H/G
            center_feature_scale=center_feature_scale)
        
        self.dw_d_dcn = self.core_op(
            channels=channels,
            kernel_size=7,
            stride=1,
            pad=9,
            dilation=3,
            group=channels,
            offset_scale=offset_scale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            dw_kernel_size=dw_kernel_size, # for InternImage-H/G
            center_feature_scale=center_feature_scale)

    def forward(self, x):
        u = x.clone()
        u = u.permute(0,3,1,2)
        # TODO add a dw 1x1 that is paralell with the attn(similar to v3+)(will be used to mimic channel attention)
        x = self.dw_dcn(x)
        x = self.dw_d_dcn(x)
        x = self.output_proj(x)
        x = x.permute(0,3,1,2)
        return x*u
        
    def _reset_parameters(self):
        self.dw_dcn._reset_parameters()
        self.dw_d_dcn._reset_parameters()
        

# TODO lag også en egen som likner mer på intern Image