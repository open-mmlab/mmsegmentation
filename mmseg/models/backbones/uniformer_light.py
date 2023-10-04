# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmseg.registry import MODELS

layer_scale = False
init_value = 1e-6
global_attn = None
token_indices = None


# code is from https://github.com/YifanXu74/Evo-ViT
def easy_gather(x, indices):
    # x => B x N x C
    # indices => B x N
    B, N, C = x.shape
    N_new = indices.shape[1]
    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
    indices = indices + offset
    # only select the informative tokens
    out = x.reshape(B * N, C)[indices.view(-1)].reshape(B, N_new, C)
    return out


# code is from https://github.com/YifanXu74/Evo-ViT
def merge_tokens(x_drop, score):
    # x_drop => B x N_drop
    # score => B x N_drop
    weight = score / torch.sum(score, dim=1, keepdim=True)
    x_drop = weight.unsqueeze(-1) * x_drop
    return torch.sum(x_drop, dim=1, keepdim=True)


def drop_path(x,
              drop_prob: float = 0.,
              training: bool = False,
              scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc
    networks, however, the original name is misleading as 'Drop Connect' is
    a different form of dropout in a separate paper...
    See discussion:
    https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    I've opted for changing the layer and argument names to 'drop path' rather
    than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks)."""

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 trade_off=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version,
        # can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # updating weight for global score
        self.trade_off = trade_off

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # update global score
        global global_attn
        tradeoff = self.trade_off
        if isinstance(global_attn, int):
            global_attn = torch.mean(attn[:, :, 0, 1:], dim=1)
        elif global_attn.shape[1] == N - 1:
            # no additional token and no pruning, update all global scores
            cls_attn = torch.mean(attn[:, :, 0, 1:], dim=1)
            global_attn = (1 - tradeoff) * global_attn + tradeoff * cls_attn
        else:
            # only update the informative tokens
            # the first one is class token
            # the last one is rrepresentative token
            cls_attn = torch.mean(attn[:, :, 0, 1:-1], dim=1)
            if self.training:
                temp_attn = (1 - tradeoff) * global_attn[:, :(
                    N - 2)] + tradeoff * cls_attn
                global_attn = torch.cat((temp_attn, global_attn[:, (N - 2):]),
                                        dim=1)
            else:
                # no use torch.cat() for fast inference
                global_attn[:, :(N - 2)] = (1 - tradeoff) * global_attn[:, :(
                    N - 2)] + tradeoff * cls_attn

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # NOTE: drop path for stochastic depth, we shall
        # see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)
        global layer_scale
        self.ls = layer_scale
        if self.ls:
            global init_value
            print(f'Use layer_scale: {layer_scale}, init_values: {init_value}')
            self.gamma_1 = nn.Parameter(
                init_value * torch.ones((1, dim, 1, 1)), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                init_value * torch.ones((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        x = x + self.pos_embed(x)
        if self.ls:
            x = x + self.drop_path(self.gamma_1 * self.conv2(
                self.attn(self.conv1(self.norm1(x)))))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(
                self.conv2(self.attn(self.conv1(self.norm1(x)))))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class EvoSABlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 prune_ratio=1,
                 trade_off=0,
                 downsample=False):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            trade_off=trade_off)
        # NOTE: drop path for stochastic depth, we shall
        # see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)
        self.prune_ratio = prune_ratio
        self.downsample = downsample
        if downsample:
            self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        global layer_scale
        self.ls = layer_scale
        if self.ls:
            global init_value
            print(f'Use layer_scale: {layer_scale}, init_values: {init_value}')
            self.gamma_1 = nn.Parameter(
                init_value * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                init_value * torch.ones(dim), requires_grad=True)
            if self.prune_ratio != 1:
                self.gamma_3 = nn.Parameter(
                    init_value * torch.ones(dim), requires_grad=True)

    def forward(self, cls_token, x):
        x = x + self.pos_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        if self.prune_ratio == 1:
            x = torch.cat([cls_token, x], dim=1)
            if self.ls:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            cls_token, x = x[:, :1], x[:, 1:]
            x = x.transpose(1, 2).reshape(B, C, H, W)
            return cls_token, x
        else:
            global global_attn, token_indices
            # calculate the number of informative tokens
            N = x.shape[1]
            N_ = int(N * self.prune_ratio)
            # sort global attention
            indices = torch.argsort(global_attn, dim=1, descending=True)

            # concatenate x, global attention and token indices => x_ga_ti
            # rearrange the tensor according to new indices
            x_ga_ti = torch.cat(
                (x, global_attn.unsqueeze(-1), token_indices.unsqueeze(-1)),
                dim=-1)
            x_ga_ti = easy_gather(x_ga_ti, indices)
            x_sorted = x_ga_ti[:, :, :-2]
            global_attn = x_ga_ti[:, :, -2]
            token_indices = x_ga_ti[:, :, -1]

            # informative tokens
            x_info = x_sorted[:, :N_]
            # merge dropped tokens
            x_drop = x_sorted[:, N_:]
            score = global_attn[:, N_:]
            #  B x N_drop x C => B x 1 x C
            rep_token = merge_tokens(x_drop, score)
            # concatenate new tokens
            x = torch.cat((cls_token, x_info, rep_token), dim=1)

            if self.ls:
                # slow update
                fast_update = 0
                tmp_x = self.attn(self.norm1(x))
                fast_update = fast_update + tmp_x[:, -1:]
                x = x + self.drop_path(self.gamma_1 * tmp_x)
                tmp_x = self.mlp(self.norm2(x))
                fast_update = fast_update + tmp_x[:, -1:]
                x = x + self.drop_path(self.gamma_2 * tmp_x)
                # fast update
                x_drop = x_drop + self.gamma_3 * fast_update.expand(
                    -1, N - N_, -1)
            else:
                # slow update
                fast_update = 0
                tmp_x = self.attn(self.norm1(x))
                fast_update = fast_update + tmp_x[:, -1:]
                x = x + self.drop_path(tmp_x)
                tmp_x = self.mlp(self.norm2(x))
                fast_update = fast_update + tmp_x[:, -1:]
                x = x + self.drop_path(tmp_x)
                # fast update
                x_drop = x_drop + fast_update.expand(-1, N - N_, -1)

            cls_token, x = x[:, :1, :], x[:, 1:-1, :]
            if self.training:
                x_sorted = torch.cat((x, x_drop), dim=1)
            else:
                x_sorted[:, N_:] = x_drop
                x_sorted[:, :N_] = x

            # recover token
            # scale for normalization
            old_global_scale = torch.sum(global_attn, dim=1, keepdim=True)
            # recover order
            indices = torch.argsort(token_indices, dim=1)
            x_ga_ti = torch.cat((x_sorted, global_attn.unsqueeze(-1),
                                 token_indices.unsqueeze(-1)),
                                dim=-1)
            x_ga_ti = easy_gather(x_ga_ti, indices)
            x_patch = x_ga_ti[:, :, :-2]
            global_attn = x_ga_ti[:, :, -2]
            token_indices = x_ga_ti[:, :, -1]
            x_patch = x_patch.transpose(1, 2).reshape(B, C, H, W)

            if self.downsample:
                # downsample global attention
                global_attn = global_attn.reshape(B, 1, H, W)
                global_attn = self.avgpool(global_attn).view(B, -1)
                # normalize global attention
                new_global_scale = torch.sum(global_attn, dim=1, keepdim=True)
                scale = old_global_scale / new_global_scale
                global_attn = global_attn * scale

            return cls_token, x_patch


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class head_embedding(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(
                out_channels // 2,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class middle_embedding(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


@MODELS.register_module()
class UniFormer_Light(nn.Module):
    """Vision Transformer A PyTorch impl of :

    `An Image is Worth 16x16 Words: Transformers for Image Recognition at
    Scale`  -     https://arxiv.org/abs/2010.11929
    """

    def __init__(self,
                 depth=[3, 4, 8, 3],
                 in_chans=3,
                 embed_dim=[64, 128, 320, 512],
                 head_dim=64,
                 mlp_ratio=[4., 4., 4., 4.],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 conv_stem=False,
                 prune_ratio=[[], [], [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                              [0.5, 0.5, 0.5]],
                 trade_off=[[], [], [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5]],
                 norm_eval=False,
                 pretrained_path=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            head_dim (int): head dimension
            mlp_ratio (list): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5
            if set representation_size (Optional[int]): enable and set
            representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.norm_eval = norm_eval
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        if conv_stem:
            self.patch_embed1 = head_embedding(
                in_channels=in_chans, out_channels=embed_dim[0])
            self.patch_embed2 = PatchEmbed(
                patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1])
            self.patch_embed3 = PatchEmbed(
                patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2])
            self.patch_embed4 = PatchEmbed(
                patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3])
        else:
            self.patch_embed1 = PatchEmbed(
                patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0])
            self.patch_embed2 = PatchEmbed(
                patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1])
            self.patch_embed3 = PatchEmbed(
                patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2])
            self.patch_embed4 = PatchEmbed(
                patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3])

        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim[2]))
        self.cls_upsample = nn.Linear(embed_dim[2], embed_dim[3])

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))
        ]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratio[0],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer) for i in range(depth[0])
        ])
        self.norm1 = norm_layer(embed_dim[0])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratio[1],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i + depth[0]],
                norm_layer=norm_layer) for i in range(depth[1])
        ])
        self.norm2 = norm_layer(embed_dim[1])
        self.blocks3 = nn.ModuleList([
            EvoSABlock(
                dim=embed_dim[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratio[2],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i + depth[0] + depth[1]],
                norm_layer=norm_layer,
                prune_ratio=prune_ratio[2][i],
                trade_off=trade_off[2][i],
                downsample=True if i == depth[2] - 1 else False)
            for i in range(depth[2])
        ])
        self.norm3 = norm_layer(embed_dim[2])
        self.blocks4 = nn.ModuleList([
            EvoSABlock(
                dim=embed_dim[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratio[3],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i + depth[0] + depth[1] + depth[2]],
                norm_layer=norm_layer,
                prune_ratio=prune_ratio[3][i],
                trade_off=trade_off[3][i]) for i in range(depth[3])
        ])
        self.norm4 = norm_layer(embed_dim[3])

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        out = []
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x_out = self.norm1(x.permute(0, 2, 3, 1))
        out.append(x_out.permute(0, 3, 1, 2).contiguous())
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x_out = self.norm2(x.permute(0, 2, 3, 1))
        out.append(x_out.permute(0, 3, 1, 2).contiguous())
        x = self.patch_embed3(x)
        # add cls_token in stage3
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        global global_attn, token_indices
        global_attn = 0
        token_indices = torch.arange(
            x.shape[2] * x.shape[3], dtype=torch.long,
            device=x.device).unsqueeze(0)
        token_indices = token_indices.expand(x.shape[0], -1)
        for blk in self.blocks3:
            cls_token, x = blk(cls_token, x)
        # upsample cls_token before stage4
        cls_token = self.cls_upsample(cls_token)
        x_out = self.norm3(x.permute(0, 2, 3, 1))
        out.append(x_out.permute(0, 3, 1, 2).contiguous())
        x = self.patch_embed4(x)
        # whether reset global attention? Now simple avgpool
        token_indices = torch.arange(
            x.shape[2] * x.shape[3], dtype=torch.long,
            device=x.device).unsqueeze(0)
        token_indices = token_indices.expand(x.shape[0], -1)
        for blk in self.blocks4:
            cls_token, x = blk(cls_token, x)
        x_out = self.norm4(x.permute(0, 2, 3, 1))
        out.append(x_out.permute(0, 3, 1, 2).contiguous())
        return tuple(out)

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
