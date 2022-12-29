import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, build_norm_layer

from ..utils import resize
from mmseg.registry import MODELS
from .aspp_head import ASPPHead, ASPPModule
from ..losses import accuracy

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, norm_cfg, proj_dim=256, proj='convmlp'):
        super(ProjectionHead, self).__init__()
        
        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                build_norm_layer(norm_cfg, dim_in)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )
            
    def forward(self, x):
        return torch.nn.functional.normalize(self.proj(x), p=2, dim=1)
    
class DepthwiseSeparableASPPModule(ASPPModule):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
    conv."""

    def __init__(self, **kwargs):
        super(DepthwiseSeparableASPPModule, self).__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.channels,
                    3,
                    dilation=dilation,
                    padding=dilation,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)


@MODELS.register_module()
class DepthwiseSeparableASPPContrastHead(ASPPHead):
    """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.
    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.
    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
    """

    def __init__(self, c1_in_channels, c1_channels, **kwargs):
        super(DepthwiseSeparableASPPContrastHead, self).__init__(**kwargs)
        assert c1_in_channels >= 0
        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations=self.dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.c1_bottleneck = None
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.proj_head = ProjectionHead(dim_in=2048, norm_cfg=self.norm_cfg)
        self.register_buffer("step", torch.zeros(1))

    def forward(self, inputs):
        """Forward function."""
        self.step+=1
        embedding = self.proj_head(inputs[-1])
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        output = self.cls_seg(output)
        return output, embedding

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """

        # HieraSeg decode_head output is: (out, embedding) :tuple, 
        # only need 'out' here.
        if isinstance(seg_logits,tuple):
            seg_logits = seg_logits[0]
        # seg_logits = resize(
        #     input=out,
        #     size=inputs.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # print('===========调用encode_decode结束=====')

        seg_logits = resize(
            input=seg_logits,
            size=batch_img_metas[0]['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits

    def losses(self, results, seg_label):
        """Compute segmentation loss."""
        seg_logit_before = results[0]
        embedding = results[1]
        loss = dict()
        seg_logit = resize(
            input=seg_logit_before,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        seg_logit_before = resize(
            input=seg_logit_before,
            scale_factor=0.5,
            mode='bilinear',
            align_corners=self.align_corners)
        loss['loss_seg'] = self.loss_decode(
            self.step,
            embedding,
            seg_logit_before,
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss
