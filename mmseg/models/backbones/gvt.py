import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, trunc_normal_init)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.runner import BaseModule, ModuleList, load_checkpoint
from torch.nn.modules.utils import _pair as to_2tuple

from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger


class Mlp(BaseModule):
    """feed forward network in Attention Module.

    Args:
        in_features (int): The feature dimension.
        hidden_features (int/None): The feature dimension of hidden layer.
            Default: None.
        out_features(int/None): he feature dimension of output layer.
            Default: None.
        act_cfg(dict): The activation config for FFNs.
            Default: dict(type='GELU').
        drop(float, optional): Dropout ratio of output. Default: 0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GroupAttention(BaseModule):
    """implementation of proposed Locally-grouped self-attention(LSA).

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 8
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: False.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.
        ws (int): the use of LSA or GSA. Default: 1.
        sr_ratio (float): kernel_size of conv. Default: 1.
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 ws=1,
                 sr_ratio=1.0):
        """ws 1 for stand attention."""
        super(GroupAttention, self).__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by ' \
                                     f'num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, H, W):
        """
        There are two implementations for this function, zero padding or mask.
        We don't observe obvious difference for both. You can choose any one,
        we recommend forward_padding because it's neat. However, the masking
        implementation is more reasonable and accurate.
        Args:
            x:
            H:
            W:

        Returns:

        """
        return self.forward_mask(x, H, W)

    def forward_mask(self, x, H, W):
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        mask = torch.zeros((1, Hp, Wp), device=x.device)
        mask[:, -pad_b:, :].fill_(1)
        mask[:, :, -pad_r:].fill_(1)

        x = x.reshape(B, _h, self.ws, _w, self.ws,
                      C).transpose(2, 3)  # B, _h, _w, ws, ws, C
        mask = mask.reshape(1, _h, self.ws, _w,
                            self.ws).transpose(2, 3).reshape(
                                1, _h * _w, self.ws * self.ws)
        attn_mask = mask.unsqueeze(2) - mask.unsqueeze(
            3)  # 1, _h*_w, ws*ws, ws*ws
        attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                          float(-1000.0)).masked_fill(
                                              attn_mask == 0, float(0.0))
        qkv = self.qkv(x).reshape(B, _h * _w, self.ws * self.ws, 3,
                                  self.num_heads, C // self.num_heads).permute(
                                      3, 0, 1, 4, 2,
                                      5)  # n_h, B, _w*_h, nhead, ws*ws, dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, _h*_w, n_head, ws*ws, dim_head
        attn = (q @ k.transpose(
            -2, -1)) * self.scale  # B, _h*_w, n_head, ws*ws, ws*ws
        attn = attn + attn_mask.unsqueeze(2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(
            attn)  # attn @v ->  B, _h*_w, n_head, ws*ws, dim_head
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws,
                                                  C)
        x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_padding(self, x, H, W):
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)
        qkv = self.qkv(x).reshape(B, _h * _w, self.ws * self.ws, 3,
                                  self.num_heads, C // self.num_heads).permute(
                                      3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws,
                                                  C)
        x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 8
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: False.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.
        sr_ratio (float): kernel_size of conv. Default: 1.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert dim % num_heads == 0, f'dim {dim} should be divided by ' \
                                     f'num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = build_conv_layer(
                dict(type='Conv2d'),
                in_channels=dim,
                out_channels=dim,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            norm_cfg = dict(type='LN')
            self.norm = build_norm_layer(norm_cfg, dim)[1]

    def forward(self, x, H, W):
        B, N, C = x.shape  # 1, 21760, 64
        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).permute(
                                         2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                    C // self.num_heads).permute(
                                        2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class TransformerEncoderLayer(BaseModule):
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
            Defalut: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (float): kernel_size of conv in Attention modules. Default: 1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: True.
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
                 sr_ratio=1.,
                 batch_first=True):
        super(TransformerEncoderLayer, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.attn = Attention(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=None,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            sr_ratio=sr_ratio)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        self.mlp = Mlp(
            in_features=embed_dims,
            hidden_features=feedforward_channels,
            act_cfg=act_cfg,
            drop=drop_rate)

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)
        ) if drop_path_rate > 0. else nn.Identity()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GroupBlock(TransformerEncoderLayer):
    """Implements one encoder layer in Twins-ALTGVT.

    Args:
       dim (int): The feature dimension.
       num_heads (int): Parallel attention heads.
       mlp_ratio (float): The hidden dimension for FFNs.
       qkv_bias (bool): enable bias for qkv if True. Default: True
       qk_scale (float | None, optional): Override default qk scale of
           head_dim ** -0.5 if set. Default: None.
       drop (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
       attn_drop (float, optional): Dropout ratio of attention weight.
           Default: 0.0
       drop_path (float): stochastic depth rate. Default 0.0.
       act_cfg (dict): The activation config for FFNs.
            Defalut: dict(type='GELU').
       norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
       sr_ratio (float): kernel_size of conv in Attention modules. Default: 1.
       ws (int): the use of LSA or GSA. Default: 1.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1,
                 ws=1):
        super(GroupBlock, self).__init__(
            dim,
            num_heads,
            mlp_ratio * dim,
            qkv_bias=qkv_bias,
            drop_rate=drop,
            attn_drop_rate=attn_drop,
            drop_path_rate=drop_path,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg)

        del self.attn
        if ws == 1:
            self.attn = Attention(dim, num_heads, qkv_bias, qk_scale,
                                  attn_drop, drop, sr_ratio)
        else:
            self.attn = GroupAttention(dim, num_heads, qkv_bias, qk_scale,
                                       attn_drop, drop, ws)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(BaseModule):
    """Image to Patch Embedding.

    Args:
       img_size (int): Input image size. Default: 224.
       patch_size (int): The patch size. Default: 16.
       in_chans (int): Number of input channels. Default: 3.
       embed_dim (int): The feature dimension. Default: 768
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % \
               patch_size[1] == 0, f'img_size {img_size} should be ' \
                                   f'divided by patch_size {patch_size}.'
        self.H, self.W = img_size[0] // patch_size[0], img_size[
            1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = build_conv_layer(
            dict(type='Conv2d'),
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size)
        norm_cfg = dict(type='LN')
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


# borrow from PVT https://github.com/whai362/PVT.git
class PyramidVisionTransformer(BaseModule):
    """Pyramid Vision Transformer.

    borrow from PVT https://github.com/whai362/PVT.git

    Args:
        img_size (int | tuple): Input image size. Default: 224.
        patch_size (int): The patch size. Default: 16.
        in_chans (int): Number of input channels. Default: 3.
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
        block_cls (BaseModule): Transformer Encoder.
            Default TransformerEncoderLayer
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
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
                 block_cls=TransformerEncoderLayer):
        super().__init__()
        print('drop_path_rate: --- ', drop_path_rate)
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embeds = ModuleList()
        self.pos_embeds = nn.ParameterList()
        self.pos_drops = ModuleList()
        self.blocks = ModuleList()

        for i in range(len(depths)):
            if i == 0:
                self.patch_embeds.append(
                    PatchEmbed(img_size, patch_size, in_chans, embed_dims[i]))
            else:
                self.patch_embeds.append(
                    PatchEmbed(img_size // patch_size // 2**(i - 1), 2,
                               embed_dims[i - 1], embed_dims[i]))
            patch_num = self.patch_embeds[-1].num_patches + 1 if i == len(
                embed_dims) - 1 else self.patch_embeds[-1].num_patches
            self.pos_embeds.append(
                nn.Parameter(torch.zeros(1, patch_num, embed_dims[i])))
            self.pos_drops.append(nn.Dropout(p=drop_rate))

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        for k in range(len(depths)):
            _block = ModuleList([
                TransformerEncoderLayer(
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
                    batch_first=True,
                    sr_ratio=sr_ratios[k]) for i in range(depths[k])
            ])
            self.blocks.append(_block)
            cur += depths[k]

        self.norm_name, norm = build_norm_layer(
            norm_cfg, embed_dims[-1], postfix=1)

        # cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[-1]))

        # classification head
        self.head = nn.Linear(
            embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        for pos_emb in self.pos_embeds:
            trunc_normal_init(pos_emb, std=.02)
        self.apply(self._init_weights)

    def reset_drop_path(self, drop_path_rate):
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        cur = 0
        for k in range(len(self.depths)):
            for i in range(self.depths[k]):
                self.blocks[k][i].drop_path.drop_prob = dpr[cur + i]
            cur += self.depths[k]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            self.apply(self._init_weights)
            logger = get_root_logger()
            load_checkpoint(
                self,
                pretrained,
                map_location='cpu',
                strict=False,
                logger=logger)
        elif pretrained is None:
            self.apply(self._init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            if i == len(self.depths) - 1:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embeds[i]
            x = self.pos_drops[i](x)
            for blk in self.blocks[i]:
                x = blk(x, H, W)
            if i < len(self.depths) - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.norm(x)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


class PosCNN(BaseModule):
    """Default Patch Embedding of CPVTV2.

    Args:
       in_chans (int): Number of input channels. Default: 3.
       embed_dim (int): The feature dimension. Default: 768.
       s (int): stride of cobnv layer. Default: 1.
    """

    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=in_chans,
                out_channels=embed_dim,
                kernel_size=3,
                stride=s,
                padding=1,
                bias=True,
                groups=embed_dim))
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class CPVTV2(PyramidVisionTransformer):
    """Use useful results from CPVT.

    PEG and GAP. Therefore, cls token is no longer required. PEG is used to
    encode the absolute position on the fly, which greatly affects the
    performance when input resolution changes during the training (such as
    segmentation, detection)

    Args:
        img_size (int | tuple): Input image size. Default: 224.
        patch_size (int): The patch size. Default: 4.
        in_chans (int): Number of input channels. Default: 3.
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
        block_cls (BaseModule): Transformer Encoder.
            Default TransformerEncoderLayer
        F4=False（bool): input features need slice.
        extra_norm（bool): add extra norm. Default False.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
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
                 block_cls=TransformerEncoderLayer,
                 F4=False,
                 extra_norm=False):
        super(CPVTV2,
              self).__init__(img_size, patch_size, in_chans, num_classes,
                             embed_dims, num_heads, mlp_ratios, qkv_bias,
                             drop_rate, attn_drop_rate, drop_path_rate,
                             norm_cfg, depths, sr_ratios, block_cls)
        self.F4 = F4
        self.extra_norm = extra_norm
        if self.extra_norm:
            self.norm_list = ModuleList()
            for dim in embed_dims:
                self.norm_list.append(build_norm_layer(norm_cfg, dim)[1])
        del self.pos_embeds
        del self.cls_token
        self.pos_block = ModuleList(
            [PosCNN(embed_dim, embed_dim) for embed_dim in embed_dims])
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

    def no_weight_decay(self):
        return set(
            ['cls_token'] +
            ['pos_block.' + n for n, p in self.pos_block.named_parameters()])

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

    def forward(self, x):
        x = self.forward_features(x)

        if self.F4:
            x = x[3:4]

        return x


class PCPVT(CPVTV2):
    """Add applicable positional encodings to CPVT. The implementation of our
    first proposed architecture: Twins-PCPVT.

    Args:
         img_size (int | tuple): Input image size. Default: 224.
         patch_size (int): The patch size. Default: 4.
         in_chans (int): Number of input channels. Default: 3.
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
         drop_path_rate (float): stochastic depth rate. Default 0.0
         norm_cfg (dict): Config dict for normalization layer.
             Default: dict(type='LN')
         depths (list): depths of each stage. Default [4, 4, 4].
         sr_ratios (list): kernel_size of conv in each Attn module in
             Transformer encoder layer. Default: [4, 2, 1].
         block_cls (BaseModule): Transformer Encoder.
            Default TransformerEncoderLayer
         F4=False（bool): input features need slice.
         extra_norm（bool): add extra norm. Default False.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=[64, 128, 256],
                 num_heads=[1, 2, 4],
                 mlp_ratios=[4, 4, 4],
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN'),
                 depths=[4, 4, 4],
                 sr_ratios=[4, 2, 1],
                 block_cls=TransformerEncoderLayer,
                 F4=False,
                 extra_norm=False):
        super(PCPVT,
              self).__init__(img_size, patch_size, in_chans, num_classes,
                             embed_dims, num_heads, mlp_ratios, qkv_bias,
                             drop_rate, attn_drop_rate, drop_path_rate,
                             norm_cfg, depths, sr_ratios, block_cls, F4,
                             extra_norm)


class ALTGVT(PCPVT):
    """Use useful results from CPVT.

    PEG and GAP. Therefore, cls token is no longer required. PEG is used to
    encode the absolute position on the fly, which greatly affects the
    performance when input resolution changes during the training (such as
    segmentation, detection)

    Args:
        img_size (int | tuple): Input image size. Default: 224.
        patch_size (int): The patch size. Default: 4.
        in_chans (int): Number of input channels. Default: 3.
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
        block_cls (BaseModule): Transformer Encoder. Default GroupBlock.
        wss=[7, 7, 7],
        F4=False（bool): input features need slice.
        extra_norm（bool): add extra norm. Default False.
        strides=(2, 2, 2)
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
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
                 block_cls=GroupBlock,
                 wss=[7, 7, 7],
                 F4=False,
                 extra_norm=False,
                 strides=(2, 2, 2)):
        super(ALTGVT,
              self).__init__(img_size, patch_size, in_chans, num_classes,
                             embed_dims, num_heads, mlp_ratios, qkv_bias,
                             drop_rate, attn_drop_rate, drop_path_rate,
                             norm_cfg, depths, sr_ratios, block_cls, F4)
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
                    dim=embed_dims[k],
                    num_heads=num_heads[k],
                    mlp_ratio=mlp_ratios[k],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_cfg=dict(type='LN'),
                    sr_ratio=sr_ratios[k],
                    ws=1 if i % 2 == 1 else wss[k]) for i in range(depths[k])
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
                        PatchEmbed(img_size, patch_size, in_chans,
                                   embed_dims[i]))
                else:
                    self.patch_embeds.append(
                        PatchEmbed(img_size // patch_size // s, strides[i - 1],
                                   embed_dims[i - 1], embed_dims[i]))
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


def _conv_filter(state_dict, patch_size=16):
    """convert patch embedding weight from manual patchify + linear proj to
    conv."""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@BACKBONES.register_module()
class pcpvt_small_v0(CPVTV2):

    def __init__(self, **kwargs):
        super(pcpvt_small_v0, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_cfg=dict(type='LN'),
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.2)


@BACKBONES.register_module()
class pcpvt_base_v0(CPVTV2):

    def __init__(self, **kwargs):
        super(pcpvt_base_v0, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_cfg=dict(type='LN'),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.2)


@BACKBONES.register_module()
class pcpvt_large(CPVTV2):

    def __init__(self, **kwargs):
        super(pcpvt_large, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_cfg=dict(type='LN'),
            depths=[3, 8, 27, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.2)


@BACKBONES.register_module()
class alt_gvt_small(ALTGVT):

    def __init__(self, **kwargs):
        super(alt_gvt_small, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 256, 512],
            num_heads=[2, 4, 8, 16],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_cfg=dict(type='LN'),
            depths=[2, 2, 10, 4],
            wss=[7, 7, 7, 7],
            sr_ratios=[8, 4, 2, 1],
            extra_norm=True,
            drop_path_rate=0.2,
        )


@BACKBONES.register_module()
class alt_gvt_base(ALTGVT):

    def __init__(self, **kwargs):
        super(alt_gvt_base, self).__init__(
            patch_size=4,
            embed_dims=[96, 192, 384, 768],
            num_heads=[3, 6, 12, 24],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_cfg=dict(type='LN'),
            depths=[2, 2, 18, 2],
            wss=[7, 7, 7, 7],
            sr_ratios=[8, 4, 2, 1],
            extra_norm=True,
            drop_path_rate=0.2,
        )


@BACKBONES.register_module()
class alt_gvt_large(ALTGVT):

    def __init__(self, **kwargs):
        super(alt_gvt_large, self).__init__(
            patch_size=4,
            embed_dims=[128, 256, 512, 1024],
            num_heads=[4, 8, 16, 32],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_cfg=dict(type='LN'),
            depths=[2, 2, 18, 2],
            wss=[7, 7, 7, 7],
            sr_ratios=[8, 4, 2, 1],
            extra_norm=True,
            drop_path_rate=0.3,
        )
