import math
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from ..builder import BACKBONES, build_backbone
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule, build_norm_layer

class BasicBlock(BaseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False,norm_cfg=dict(type='SyncBN'), init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg=init_cfg)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,bias=False,padding=1)    
        self.bn1 =build_norm_layer(norm_cfg, planes)[1]   
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,bias=False,padding=1)     
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]  
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

class Bottleneck(BaseModule):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True,norm_cfg=dict(type='SyncBN'), init_cfg=None):
        super(Bottleneck, self).__init__(init_cfg=init_cfg)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]   
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 =  build_norm_layer(norm_cfg, planes)[1]   
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = build_norm_layer(norm_cfg,planes * self.expansion )[1]  
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class BNReLuConv(BaseModule):
    def __init__(self, inplanes, planes, kernel_size,padding=0,stride=1,norm_cfg=dict(type='BN'), init_cfg=None):
        super(BNReLuConv, self).__init__(init_cfg=init_cfg)
        self.f=nn.Sequential(
            build_norm_layer(norm_cfg, inplanes)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,bias=False,padding=padding)
            )
    def forward(self,x):
        x=self.f(x)
        return x

class DAPPM(BaseModule):
    def __init__(self, inplanes, branch_planes, out_planes, norm_cfg=dict(type='SyncBN'), align_corners=False, init_cfg=dict(type='Kaiming', distribution='normal')):
        super(DAPPM, self).__init__(init_cfg)

    # scale
        self.scale_kernel_sizes=[5,9,17]
        self.scale_strides=[2,4,8]
        self.scale_paddings=[2,4,8]
        
        self.scale= nn.ModuleList()
        self.scale.append(BNReLuConv(inplanes,branch_planes,1,norm_cfg=norm_cfg))

        for v in range(3):
            self.scale.append(nn.Sequential(
                nn.AvgPool2d(kernel_size=self.scale_kernel_sizes[v], stride=self.scale_strides[v], padding=self.scale_paddings[v]),
                BNReLuConv(inplanes,branch_planes,1,norm_cfg=norm_cfg),
                )
                )
        self.scale.append(nn.Sequential(
           nn.AdaptiveAvgPool2d((1, 1)),
            BNReLuConv(inplanes,branch_planes,1,norm_cfg=norm_cfg),
           ))

       # process
        self.process= nn.ModuleList()
        for v in range(4):
            self.process.append(BNReLuConv(branch_planes,branch_planes,3,padding=1,norm_cfg=norm_cfg))
      
        self.compression = nn.Sequential(
            BNReLuConv(branch_planes*5,out_planes,1,norm_cfg=norm_cfg),
        )

        self.shortcut = nn.Sequential(
            BNReLuConv(inplanes,out_planes,1,norm_cfg=norm_cfg),
        )
        self.align_corners=align_corners

                
    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        x_list=[self.scale[0](x)]
        for v in range(4):
            x_list.append(
                self.process[v](
                F.interpolate(self.scale[v+1](x),size=[height, width],mode='bilinear',align_corners=self.align_corners)+x_list[-1]
                )
                )
            
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out

def _make_layer(  block, inplanes, planes, blocks, stride=1,norm_cfg=dict(type='SyncBN')):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
               build_norm_layer(norm_cfg,planes * block.expansion )[1] ,
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample,norm_cfg=norm_cfg))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True,norm_cfg=norm_cfg))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False,norm_cfg=norm_cfg))

        return nn.Sequential(*layers)

class layer_with_fusion(BaseModule):
    def __init__(self,block_type,block_nums,
                 main_stride,main_inplanes,main_outplanes,
                 sub_stride,sub_inplanes,sub_outplanes,
                 down_type='one',
                 norm_cfg=dict(type='SyncBN'),
                 align_corners=False, init_cfg=None):
        super(layer_with_fusion, self).__init__(init_cfg=init_cfg)
        self.align_corners=align_corners
        self.main_layer =  _make_layer(block_type, main_inplanes, main_outplanes, block_nums, stride=main_stride, norm_cfg=norm_cfg)
        self.sub_layer =  _make_layer(block_type,sub_inplanes, sub_outplanes, block_nums, stride=sub_stride, norm_cfg=norm_cfg)
        self.compression  = nn.Sequential(
                                       nn.Conv2d(main_outplanes, sub_outplanes, kernel_size=1, stride=1, padding=0, bias=False),
                                       build_norm_layer(norm_cfg, sub_outplanes)[1],
                                       )
        if down_type=='two':
            self.down  = nn.Sequential(
                                   nn.Conv2d(sub_outplanes,sub_outplanes*2, kernel_size=3, stride=2, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, sub_outplanes*2)[1],
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(sub_outplanes*2, main_outplanes, kernel_size=3, stride=2, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, main_outplanes)[1],
                                   )
            self.upsample=nn.Upsample(scale_factor=4,mode='bilinear',align_corners=self.align_corners)
        if down_type=='one':
            self.down  = nn.Sequential(
                                   nn.Conv2d( sub_outplanes, main_outplanes, kernel_size=3, stride=2, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, main_outplanes)[1],
                                   )
            self.upsample=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=self.align_corners)
           
        self.relu= nn.ReLU(inplace=False)
    def forward(self,x,x_):
        x=self.main_layer(self.relu(x))
        out_=self.sub_layer(self.relu(x_))
        out=x+self.down(self.relu(out_))
        out_=out_+self.upsample   ( self.compression (self.relu(x)) ) 
        return out,out_
        

@BACKBONES.register_module()
class DualResNet( BaseModule):
    
    def __init__(self, block=BasicBlock, layers=(3, 4, (3,3), 3),  planes=64, spp_planes=128, head_planes=256,norm_cfg=dict(type='SyncBN'),align_corners=False,init_cfg=None):
        super(DualResNet, self).__init__(init_cfg) 

        highres_planes = planes * 2
       
        self.layers=layers
        self.align_corners= align_corners 

        self.conv1 =  nn.Sequential(
                          nn.Conv2d(3,planes,kernel_size=3, stride=2, padding=1),
                          build_norm_layer(norm_cfg,planes )[1]  ,
                          nn.ReLU(inplace=True),
                          nn.Conv2d(planes,planes,kernel_size=3, stride=2, padding=1),
                          build_norm_layer(norm_cfg,planes )[1]   ,
                          nn.ReLU(inplace=True),
                      )

        self.relu = nn.ReLU(inplace=False)
        self.layer1 = _make_layer(block, planes, planes, layers[0],norm_cfg=norm_cfg)
        self.layer2 = _make_layer(block, planes, planes * 2, layers[1], stride=2,norm_cfg=norm_cfg)
        self.layer3_fuison=layer_with_fusion( BasicBlock,block_nums=self.layers[2][0],
                     main_stride=2,main_inplanes=planes * 2,main_outplanes=planes * 4,
                     sub_stride=1,sub_inplanes=planes * 2,sub_outplanes=planes * 2,
                     down_type='one',
                     norm_cfg=norm_cfg,
                     align_corners=align_corners)
        if len(self.layers[2])==2:
            self.layer3_fuison_2=layer_with_fusion( BasicBlock,block_nums=self.layers[2][1],
                 main_stride=1,main_inplanes=planes * 4,main_outplanes=planes * 4,
                 sub_stride=1,sub_inplanes=planes * 2,sub_outplanes=planes * 2,
                 down_type='one',
                 norm_cfg=norm_cfg,
                 align_corners=align_corners)
                 
        self.layer4_fuison=layer_with_fusion( BasicBlock,block_nums=self.layers[3],
                 main_stride=2,main_inplanes= planes * 4,main_outplanes=planes * 8,
                 sub_stride=1,sub_inplanes=highres_planes,sub_outplanes=highres_planes,
                 down_type='two',
                 norm_cfg=norm_cfg,
                 align_corners=align_corners)

        self.layer5_ = _make_layer(Bottleneck, highres_planes, highres_planes, 1, norm_cfg=norm_cfg)
        self.layer5 = _make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2, norm_cfg=norm_cfg)

        self.spp = DAPPM(planes * 16,  spp_planes, planes * 4, norm_cfg=norm_cfg,align_corners= align_corners)
    
        self.extra_process = nn.Sequential(
                          build_norm_layer(norm_cfg,highres_planes )[1]   , 
                          nn.ReLU(inplace=True),
                         )
        self.final_process = nn.Sequential(
                          build_norm_layer(norm_cfg,planes * 4 )[1]   , 
                          nn.ReLU(inplace=True),
                         )
       

    def forward(self, x):

        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        
        x = self.conv1(x)
        x = self.layer1(x)
        
        x = self.layer2(self.relu(x))
  
        
        if len(self.layers[2])==1:
            x,x_=self.layer3_fuison(x,x)
        if len(self.layers[2])==2:
            x,x_=self.layer3_fuison(x,x)
            x,x_=self.layer3_fuison_2(x,x_)
        temp =  x_ 

        
        x,x_=self.layer4_fuison(x,x_)
       
        x_ = self.layer5_(self.relu(x_))
        x = F.interpolate(
            self.spp(self.layer5(self.relu(x))),
            size=[height_output, width_output],
            mode='bilinear',  align_corners=self.align_corners
        )

        out = x + x_

        out = self.final_process(out)

        
        out_extra = self.extra_process (temp)
        return [out_extra, out]
    

