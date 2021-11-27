import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn import build_norm_layer
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import ModuleList

from mmseg.models.backbones.vit import TransformerEncoderLayer
from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class SegmenterMaskTransformerHead(BaseDecodeHead):

    def __init__(
            self,
            in_channels,
            num_layers,
            num_heads,
            embed_dims,
            mlp_ratio=4,
            drop_path_rate=0.1,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            num_fcs=2,
            qkv_bias=True,
            act_cfg=dict(type='GELU'),
            norm_cfg=dict(type='LN'),
            **kwargs,
    ):
        super(SegmenterMaskTransformerHead, self).__init__(
            in_channels=in_channels,
            init_cfg=dict(type='TruncNormal', std=0.02),
            **kwargs,
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    batch_first=True,
                ))

        self.proj_dec = nn.Linear(in_channels, embed_dims)

        self.cls_emb = nn.Parameter(
            torch.randn(1, self.num_classes, embed_dims))
        self.proj_patch = nn.Parameter(torch.randn(embed_dims, embed_dims))
        self.proj_classes = nn.Parameter(torch.randn(embed_dims, embed_dims))

        _, self.decoder_norm = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        _, self.mask_norm = build_norm_layer(
            norm_cfg, self.num_classes, postfix=2)

        trunc_normal_(self.cls_emb, std=0.02)
        trunc_normal_(self.proj_patch, std=0.02)
        trunc_normal_(self.proj_classes, std=0.02)

        delattr(self, 'conv_seg')

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        GS = x.shape[-1]
        x = rearrange(x, 'b n h w -> b (h w) n')

        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for layer in self.layers:
            x = layer(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, :-self.num_classes], x[:,
                                                            -self.num_classes:]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, 'b (h w) n -> b n h w', h=int(GS))

        return masks
