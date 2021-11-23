import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer, trunc_normal_init
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN
from mmcv.runner import BaseModule, ModuleList

from mmseg.models.backbones.mit import EfficientMultiheadAttention
from mmseg.models.backbones.swin import WindowMSA
from mmseg.models.builder import BACKBONES
from ..utils.embed import PatchEmbed


class GlobalSubsampledAttention(EfficientMultiheadAttention):
    """global sub-sampled attention (Spatial Reduction Attention)
        This module is modified from EfficientMultiheadAttention
        which is a module from mmseg.models.backbones.mit.py
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut. Default: None.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        qkv_bias (bool): enable bias for qkv if True. Default: True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Spatial Reduction
            Attention of PVT. Default: 1.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 batch_first=True,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1):
        super(GlobalSubsampledAttention, self).__init__(
            embed_dims,
            num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            dropout_layer=dropout_layer,
            init_cfg=None,
            batch_first=batch_first,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio)


class PCPVTEncoderLayer(BaseModule):
    """Implements one encoder layer in Twins-PCPVT.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): enable bias for qkv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (float): kernel_size of conv in Attention modules. Default: 1.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1.):
        super(PCPVTEncoderLayer, self).__init__()

        self.norm1 = build_norm_layer(norm_cfg, embed_dims, postfix=1)[1]

        self.attn = GlobalSubsampledAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims, postfix=2)[1]

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=False,
            init_cfg=None)

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)
        ) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), (H, W), identity=0.))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class LocallygroupedSelfAttention(WindowMSA):
    """Locally-grouped self-attention(LSA).

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 8
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: False.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        window_size(int): the use of LSA or GSA. Default: 1.
        forward padding
    """

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 window_size=1):
        """window_size 1 for stand attention."""
        super(LocallygroupedSelfAttention, self).__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=[window_size] * 2,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate)

        del self.relative_position_bias_table
        del self.window_size
        self.window_size = window_size
        assert embed_dims % num_heads == 0, f'dim {embed_dims} should be ' \
                                            f'divided by num_heads ' \
                                            f'{num_heads}.'

    def forward(self, x, hw_shape, identity=None):
        B, N, C = x.shape
        H, W = hw_shape
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of Local-groups
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        # calculate attention mask for LSA
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.window_size, Wp // self.window_size
        mask = torch.zeros((1, Hp, Wp), device=x.device)
        mask[:, -pad_b:, :].fill_(1)
        mask[:, :, -pad_r:].fill_(1)

        x = x.reshape(B, _h, self.window_size, _w, self.window_size,
                      C).transpose(2, 3)
        mask = mask.reshape(1, _h, self.window_size, _w,
                            self.window_size).transpose(2, 3).reshape(
                                1, _h * _w,
                                self.window_size * self.window_size)
        attn_mask = mask.unsqueeze(2) - mask.unsqueeze(3)
        attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                          float(-1000.0)).masked_fill(
                                              attn_mask == 0, float(0.0))

        # calculate multi-head self-attention
        qkv = self.qkv(x).reshape(B, _h * _w,
                                  self.window_size * self.window_size, 3,
                                  self.num_heads, C // self.num_heads).permute(
                                      3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + attn_mask.unsqueeze(2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.window_size,
                                                  self.window_size, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.window_size,
                                         _w * self.window_size, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SVTEncoderLayer(PCPVTEncoderLayer):
    """Implements one encoder layer in Twins-SVT.

    Args:
        embed_dims (int): The feature dimension.
       num_heads (int): Parallel attention heads.
       mlp_ratio (float): The hidden dimension for FFNs.
       qkv_bias (bool): enable bias for qkv if True. Default: True
       qk_scale (float | None, optional): Override default qk scale of
           head_dim ** -0.5 if set. Default: None.
       drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
       attn_drop_rate (float, optional): Dropout ratio of attention weight.
           Default: 0.0
       drop_path_rate (float): stochastic depth rate. Default 0.0.
       act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
       norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
       sr_ratio (float): kernel_size of conv in Attention modules. Default: 1.
       window_size (int): the use of LSA or GSA. Default: 1.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1.,
                 window_size=1):
        super(SVTEncoderLayer, self).__init__(
            embed_dims,
            num_heads,
            mlp_ratio * embed_dims,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio)

        if window_size != 1:
            self.attn = LocallygroupedSelfAttention(embed_dims, num_heads,
                                                    qkv_bias, qk_scale,
                                                    attn_drop_rate, drop_rate,
                                                    window_size)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), (H, W), identity=0.))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class ConditionalPositionEncoding(BaseModule):
    """Default Patch Embedding of CPVTV2.

    Args:
       in_channels (int): Number of input channels. Default: 3.
       embed_dim (int): The feature dimension. Default: 768.
       stride (int): stride of cobnv layer. Default: 1.
    """

    def __init__(self, in_channels, embed_dim=768, stride=1):
        super(ConditionalPositionEncoding, self).__init__()
        self.proj = nn.Sequential(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=True,
                groups=embed_dim))
        self.stride = stride

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.stride == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x


@BACKBONES.register_module()
class PCPVT(BaseModule):
    """The backbone of Twins-PCPVT.

    Args:
        img_size (int | tuple): Input image size. Default: 224.
        patch_size (int): The patch size. Default: 4.
        in_channels (int): Number of input channels. Default: 3.
        num_classes (int): Number of num_classes. Default: 1000
        embed_dims (list): embedding dimension. Default: [64, 128, 256, 512].
        num_heads (int): number of attention heads. Default: [1, 2, 4, 8].
        mlp_ratios (int): ratio of mlp hidden dim to embedding dim.
            Default: [4, 4, 4, 4].
        qkv_bias (bool): enable bias for qkv if True. Default: False.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        depths (list): depths of each stage. Default [3, 4, 6, 3]
        sr_ratios (list): kernel_size of conv in each Attn module in
            Transformer encoder layer. Default: [8, 4, 2, 1].
        input_features_slice（bool): input features need slice. Default False.
        extra_norm（bool): add extra norm. Default False.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_channels=3,
                 num_classes=1000,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN'),
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 input_features_slice=False,
                 extra_norm=False,
                 **kwargs):
        super(PCPVT, self).__init__()
        print('drop_path_rate: --- ', drop_path_rate)
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embeds = ModuleList()
        self.pos_drops = ModuleList()
        self.blocks = ModuleList()

        for i in range(len(depths)):
            if i == 0:
                input_size = img_size
                self.patch_embeds.append(
                    PatchEmbed(
                        in_channels=in_channels,
                        embed_dims=embed_dims[i],
                        conv_type='Conv2d',
                        kernel_size=patch_size,
                        stride=patch_size,
                        padding='corner',
                        norm_cfg=norm_cfg,
                        input_size=input_size,
                        init_cfg=None))
            else:
                patch_size = 2
                input_size = img_size // patch_size // 2**(i - 1)
                self.patch_embeds.append(
                    PatchEmbed(
                        in_channels=embed_dims[i - 1],
                        embed_dims=embed_dims[i],
                        conv_type='Conv2d',
                        kernel_size=patch_size,
                        stride=patch_size,
                        padding='corner',
                        norm_cfg=norm_cfg,
                        input_size=input_size,
                        init_cfg=None))

            self.pos_drops.append(nn.Dropout(p=drop_rate))

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        for k in range(len(depths)):
            _block = ModuleList([
                PCPVTEncoderLayer(
                    embed_dims=embed_dims[k],
                    num_heads=num_heads[k],
                    feedforward_channels=mlp_ratios[k] * embed_dims[k],
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[cur + i],
                    num_fcs=2,
                    qkv_bias=qkv_bias,
                    act_cfg=dict(type='GELU'),
                    norm_cfg=dict(type='LN'),
                    sr_ratio=sr_ratios[k]) for i in range(depths[k])
            ])
            self.blocks.append(_block)
            cur += depths[k]

        self.norm_name, norm = build_norm_layer(
            norm_cfg, embed_dims[-1], postfix=1)

        self.input_features_slice = input_features_slice
        self.extra_norm = extra_norm
        if self.extra_norm:
            self.norm_list = ModuleList()
            for dim in embed_dims:
                self.norm_list.append(build_norm_layer(norm_cfg, dim)[1])

        self.pos_block = ModuleList([
            ConditionalPositionEncoding(embed_dim, embed_dim)
            for embed_dim in embed_dims
        ])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        import math
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def forward(self, x):
        outputs = list()

        B = x.shape[0]

        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            x = self.pos_drops[i](x)
            for j, blk in enumerate(self.blocks[i]):
                x = blk(x, H, W)
                if j == 0:
                    x = self.pos_block[i](x, H, W)
            if self.extra_norm:
                x = self.norm_list[i](x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            outputs.append(x)

        if self.input_features_slice:
            outputs = outputs[3:4]

        return outputs


@BACKBONES.register_module()
class SVT(PCPVT):
    """The backbone of Twins-SVT.

    Args:
        img_size (int | tuple): Input image size. Default: 224.
        patch_size (int): The patch size. Default: 4.
        in_channels (int): Number of input channels. Default: 3.
        num_classes (int): Number of num_classes. Default: 1000
        embed_dims (list): embedding dimension. Default: [64, 128, 256].
        num_heads (int): number of attention heads. Default: [1, 2, 4].
        mlp_ratios (int): ratio of mlp hidden dim to embedding dim.
            Default: [4, 4, 4].
        qkv_bias (bool): enable bias for qkv if True. Default: False.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.2.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        depths (list): depths of each stage. Default [4, 4, 4].
        sr_ratios (list): kernel_size of conv in each Attn module in
            Transformer encoder layer. Default: [4, 2, 1].
        block_cls (BaseModule): Transformer Encoder. Default SVTEncoderLayer.
        wss=[7, 7, 7],
        input_features_slice（bool): input features need slice. Default False.
        extra_norm（bool): add extra norm. Default False.
        strides=(2, 2, 2)
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_channels=3,
                 num_classes=1000,
                 embed_dims=[64, 128, 256],
                 num_heads=[1, 2, 4],
                 mlp_ratios=[4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_cfg=dict(type='LN'),
                 depths=[4, 4, 4],
                 sr_ratios=[4, 2, 1],
                 block_cls=SVTEncoderLayer,
                 wss=[7, 7, 7],
                 input_features_slice=False,
                 extra_norm=False,
                 strides=(2, 2, 2),
                 **kwargs):
        super(SVT,
              self).__init__(img_size, patch_size, in_channels, num_classes,
                             embed_dims, num_heads, mlp_ratios, qkv_bias,
                             drop_rate, attn_drop_rate, drop_path_rate,
                             norm_cfg, depths, sr_ratios, input_features_slice)
        del self.blocks
        self.wss = wss
        self.extra_norm = extra_norm
        self.strides = strides
        if self.extra_norm:
            self.norm_list = ModuleList()
            for dim in embed_dims:
                self.norm_list.append(build_norm_layer(norm_cfg, dim)[1])
        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        self.blocks = ModuleList()
        for k in range(len(depths)):
            _block = ModuleList([
                block_cls(
                    embed_dims=embed_dims[k],
                    num_heads=num_heads[k],
                    mlp_ratio=mlp_ratios[k],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + i],
                    norm_cfg=dict(type='LN'),
                    sr_ratio=sr_ratios[k],
                    window_size=1 if i % 2 == 1 else wss[k])
                for i in range(depths[k])
            ])

            self.blocks.append(_block)
            cur += depths[k]

        if strides != (2, 2, 2):
            del self.patch_embeds
            self.patch_embeds = ModuleList()
            s = 1
            for i in range(len(depths)):
                if i == 0:
                    self.patch_embeds.append(
                        PatchEmbed(
                            in_channels=in_channels,
                            embed_dims=embed_dims[i],
                            conv_type='Conv2d',
                            kernel_size=patch_size,
                            stride=patch_size,
                            padding='corner',
                            norm_cfg=norm_cfg,
                            input_size=img_size,
                            init_cfg=None))
                else:
                    self.patch_embeds.append(
                        PatchEmbed(
                            in_channels=embed_dims[i - 1],
                            embed_dims=embed_dims[i],
                            conv_type='Conv2d',
                            kernel_size=strides[i - 1],
                            stride=strides[i - 1],
                            padding='corner',
                            norm_cfg=norm_cfg,
                            input_size=img_size // patch_size // s,
                            init_cfg=None))
                s = s * strides[i - 1]

        self.apply(self._init_weights)

    def forward_features(self, x):
        outputs = list()

        B = x.shape[0]

        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            x = self.pos_drops[i](x)
            for j, blk in enumerate(self.blocks[i]):
                x = blk(x, H, W)
                if j == 0:
                    x = self.pos_block[i](x, H, W)
            if self.extra_norm:
                x = self.norm_list[i](x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outputs.append(x)

        return outputs
