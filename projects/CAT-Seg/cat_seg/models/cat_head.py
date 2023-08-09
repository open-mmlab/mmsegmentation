# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


class UpBlock(nn.Module):
    """Upsample Block with two consecutive convolution layers."""

    def __init__(self, in_channels, out_channels, guidance_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels - guidance_channels,
            kernel_size=2,
            stride=2)
        self.conv1 = ConvModule(
            in_channels,
            out_channels,
            3,
            padding=1,
            bias=False,
            norm_cfg=dict(type='GN', num_groups=out_channels // 16))
        self.conv2 = ConvModule(
            out_channels,
            out_channels,
            3,
            padding=1,
            bias=False,
            norm_cfg=dict(type='GN', num_groups=out_channels // 16))

    def forward(self, x, guidance=None):
        """Forward function with visual guidance."""
        x = self.up(x)
        if guidance is not None:
            T = x.size(0) // guidance.size(0)
            # guidance = repeat(guidance, "B C H W -> (B T) C H W", T=T)
            guidance = guidance.repeat(T, 1, 1, 1)
            x = torch.cat([x, guidance], dim=1)
        x = self.conv1(x)

        return self.conv2(x)


@MODELS.register_module()
class CATSegHead(BaseDecodeHead):
    """CATSeg Head.

    This segmentation head is the mmseg implementation of
    `CAT-Seg <https://arxiv.org/abs/2303.11797>`_.

    Args:
        embed_dims (int): The number of input dimensions.
        decoder_dims (list): The number of decoder dimensions.
        decoder_guidance_proj_dims (list): The number of appearance
            guidance dimensions.
        init_cfg
    """

    def __init__(self,
                 embed_dims=128,
                 decoder_dims=(64, 32),
                 decoder_guidance_dims=(256, 128),
                 decoder_guidance_proj_dims=(32, 16),
                 **kwargs):
        super().__init__(**kwargs)
        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    dec_dims,
                    dec_dims_proj,
                    kernel_size=3,
                    stride=1,
                    padding=1),
                nn.ReLU(),
            ) for dec_dims, dec_dims_proj in zip(decoder_guidance_dims,
                                                 decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None

        self.decoder1 = UpBlock(embed_dims, decoder_dims[0],
                                decoder_guidance_proj_dims[0])
        self.decoder2 = UpBlock(decoder_dims[0], decoder_dims[1],
                                decoder_guidance_proj_dims[1])
        self.conv_seg = nn.Conv2d(
            decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        """Forward function.

        Args:
            inputs (dict): Input features including the following features,
                corr_embed: aggregated correlation embeddings.
                appearance_feats: decoder appearance feature guidance.
        """
        # decoder guidance projection
        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [
                proj(g) for proj, g in zip(self.decoder_guidance_projection,
                                           inputs['appearance_feats'])
            ]

        # decoder layers
        B = inputs['corr_embed'].size(0)
        corr_embed = inputs['corr_embed'].transpose(1, 2).flatten(0, 1)
        corr_embed = self.decoder1(corr_embed, projected_decoder_guidance[0])
        corr_embed = self.decoder2(corr_embed, projected_decoder_guidance[1])

        output = self.cls_seg(corr_embed)

        # rearrange the output to (B, T, H, W)
        H_ori, W_ori = output.shape[-2:]
        output = output.reshape(B, -1, H_ori, W_ori)
        return output
