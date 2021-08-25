import math
import torch
import torch.nn as nn

from mmcv.cnn import  trunc_normal_init
from mmcv.runner.base_module import BaseModule, ModuleList

from ..builder import HEADS
from ..utils import TransformerEncoderLayer
from .decode_head import BaseDecodeHead

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_init(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

@HEADS.register_module()
class LinearTransformerHead(BaseDecodeHead):
    """ Segmenter-Linear
    A PyTorch implement of : `Segmenter: Transformer for Semantic Segmentation`
        https://arxiv.org/abs/2105.05633
        
    Inspiration from
        https://github.com/rstrudel/segmenter
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.head = nn.Linear(self.in_channels, self.num_classes)
        self.apply(init_weights)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = self.head(x)
        output = x.view(-1, H, W, self.num_classes).permute(0, 3, 1, 2).contiguous()

        return output

@HEADS.register_module()
class MaskTransformerHead(BaseDecodeHead):
    """ Segmenter-Mask
    A PyTorch implement of : `Segmenter: Transformer for Semantic Segmentation`
        https://arxiv.org/abs/2105.05633

    Inspiration from
        https://github.com/rstrudel/segmenter

    Args:
        num_layers (int): depth of transformer in decoder part. Default: 12.
        num_heads (int): Parallel attention heads. Default: 12.
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim(self.channels).
            Default: 4.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): enable bias for qkv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Defalut: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: True.
    """
    def __init__(
        self,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        num_fcs=2,
        qkv_bias=True,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN'),
        batch_first=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.scale = self.channels ** -0.5

        self.proj = nn.Linear(self.in_channels, self.channels)
        self.cls_embed = nn.Parameter(torch.randn(1, self.num_classes, self.channels))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.blocks = ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                TransformerEncoderLayer(
                    embed_dims=self.channels,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * self.channels,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    batch_first=True))

        self.proj_patch = nn.Parameter(self.scale * torch.randn(self.channels, self.channels))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(self.channels, self.channels))

        self.decoder_norm = nn.LayerNorm(self.channels)
        self.mask_norm = nn.LayerNorm(self.num_classes)

        self.apply(init_weights)
        trunc_normal_init(self.cls_embed, std=0.02)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)

        x = self.proj(x)
        cls_embed = self.cls_embed.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_embed), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.num_classes], x[:, -self.num_classes :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        output = masks.view(-1, H, W, self.num_classes).permute(0, 3, 1, 2).contiguous()

        return output
