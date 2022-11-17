import torch
import torch.nn as nn

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@HEADS.register_module()
class DestHead(BaseDecodeHead):
    def __init__(self, segm=True, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_channels)

        self.fuse_conv1 = nn.Sequential(nn.Conv2d(self.in_channels[-1], self.in_channels[-1], 1), nn.ReLU(inplace=True))
        self.fuse_conv2 = nn.Sequential(nn.Conv2d(self.in_channels[-2], self.in_channels[-2], 1), nn.ReLU(inplace=True))
        self.fuse_conv3 = nn.Sequential(nn.Conv2d(self.in_channels[-3], self.in_channels[-3], 1), nn.ReLU(inplace=True))
        self.fuse_conv4 = nn.Sequential(nn.Conv2d(self.in_channels[-4], self.in_channels[-4], 1), nn.ReLU(inplace=True))

        self.upsample = nn.ModuleList([nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'))]*len(self.in_channels))

        self.fused_1 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(self.in_channels[-1], self.in_channels[-1], 3), nn.ReLU(inplace=True))
        self.fused_2 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(self.in_channels[-2] + self.in_channels[-1], self.in_channels[-2], 3), nn.ReLU(inplace=True))
        self.fused_3 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(self.in_channels[-3] + self.in_channels[-2], self.in_channels[-3], 3), nn.ReLU(inplace=True))
        self.fused_4 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(self.in_channels[-4] + self.in_channels[-3], self.in_channels[-4], 3), nn.ReLU(inplace=True))
        
        self.conv_seg = nn.Conv2d(self.in_channels[-4], self.num_classes, kernel_size=1)

    def dest_decoder(self, lay_out):
        lay_out = lay_out[0]
        fused_1 = self.fuse_conv1(lay_out[-1])
        fused_1 = self.upsample[-1](fused_1)
        fused_1 = self.fused_1(fused_1)
        fused_2 = torch.cat([fused_1, self.fuse_conv2(lay_out[-2])], 1)

        fused_2 = self.upsample[-2](fused_2)
        fused_2 = self.fused_2(fused_2)
        fused_3 = torch.cat([fused_2, self.fuse_conv3(lay_out[-3])], 1)

        fused_3 = self.upsample[-3](fused_3)
        fused_3 = self.fused_3(fused_3)
        fused_4 = torch.cat([fused_3, self.fuse_conv4(lay_out[-4])], 1)

        fused_4 = self.upsample[-4](fused_4)
        fused_4 = self.fused_4(fused_4)
        
        return self.conv_seg(fused_4)
        
    def forward(self, x):
        return self.dest_decoder(x)
            

