"""Modified from https://github.com/rwightman/pytorch-image-
models/blob/master/timm/models/vision_transformer.py."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (Conv2d, Linear, build_activation_layer, build_norm_layer,
                      constant_init, kaiming_init, normal_init, xavier_init)
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmseg.utils import get_root_logger
from ..builder import BACKBONES


class Mlp(nn.Module):
    """MLP layer for Encoder block
    Args:
        in_features(int): Input dimension for the first fully
            connected layer.
        hidden_features(int): Output dimension for the first fully
            connected layer.
        out_features(int): Output dementsion for the second fully
            connected layer.
        act_cfg(dict): Config dict for activation layer.
            Default: dict(type='GELU').
        drop(float): Drop rate for the dropout layer. Dropout rate has
            to be between 0 and 1. Default: 0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """ Attention layer for Encoder block
    Args:
        dim (int): Dimension for the input vector.
        num_heads (int): Number of parallel attention heads.
        qkv_bias (bool): Enable bias for qkv if True. Default: False.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        attn_drop (float): Drop rate for attention output weights.
            Default: 0.
        proj_drop (float): Drop rate for output weights. Default: 0.
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads,
                                  c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Implements encoder block with residual connection.

    Args:
        dim (int): The feature dimension.
        num_heads (int): Number of parallel attention heads.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop (float): Drop rate for mlp output weights. Default: 0.
        attn_drop (float): Drop rate for attention output weights.
            Default: 0.
        proj_drop (float): Drop rate for attn layer output weights.
            Default: 0.
        drop_path (float): Drop rate for drop_path layer(Not implemented).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN', requires_grad=True).
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 proj_drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN')):
        super(Block, self).__init__()
        _, self.norm1 = build_norm_layer(norm_cfg, dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop,
                              proj_drop)
        _, self.norm2 = build_norm_layer(norm_cfg, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_cfg=act_cfg,
            drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    Args:
        img_size (int): Width and height for input image (img_size x img_size).
            default: 224.
        patch_size (int): Width and height for a patch.
            default: 16.
        in_channels (int): Input channels for images. Default: 3.
        embed_dim (int): The embedding dimension. Default: 768.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768):
        super(PatchEmbed, self).__init__()
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        elif isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            raise TypeError('img_size must be type of int or tuple')
        h, w = self.img_size
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (h // patch_size) * (w // patch_size)
        self.proj = Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


@BACKBONES.register_module()
class VisionTransformer(nn.Module):
    """VisionTransformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for
        Image Recognition at Scale` - https://arxiv.org/abs/2010.11929
    Args:
        img_size (tuple): input image size. Default: (224, 224）.
        pretrain_img_size (int, tuple): pretrained model img size. Default 224.
        patch_size (int, tuple): patch size. Default: 16.
        in_channels (int): number of input channels. Default: 3.
        embed_dim (int): embedding dimension. Default: 768.
        depth (int): depth of transformer. Default: 12.
        num_heads (int): number of attention heads. Default: 12.
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): enable bias for qkv if True. Default: True.
        qk_scale (float): override default qk scale of head_dim ** -0.5 if set.
        representation_size (Optional[int]): enable and set representation
            layer (pre-logits) to this value if set.
        drop_rate (float): dropout rate. Default: 0.
        attn_drop_rate (float): attention dropout rate. Default: 0.
        drop_path_rate (float): stochastic depth rate. Default: 0.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='GELU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): (Not Implement) Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        weight_init: (str): weight init mode.
    """

    def __init__(self,
                 img_size=(224, 224),
                 patch_size=16,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 representation_size=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 norm_eval=False,
                 with_cp=False):
        super(VisionTransformer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.pretrain_img_size = pretrain_img_size
        self.features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim)
        num_patches = (pretrain_img_size // patch_size)**2

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg) for i in range(depth)
        ])
        _, self.norm = build_norm_layer(norm_cfg, embed_dim)

        self.norm_eval = norm_eval
        self.with_cp = with_cp

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            state_dict = load_checkpoint(
                self, pretrained, strict=False, logger=logger)
            if 'pos_embed' in state_dict.keys(
            ) and state_dict['pos_embed'].shape != self.pos_embed.shape:
                # Resize the pos_embed to pretrained model' pos_embed.
                # self.pos_embed = nn.Parameter(
                #     torch.zeros(state_dict['pos_embed'].shape))
                self.pos_embed = nn.Parameter(state_dict['pos_embed'])
                logger.info(
                    msg='Resize the pos_embed to pretrained model pos_embed')

                self.pos_embed = nn.Parameter(self.pos_embed[:, 1:, :])
                logger.info(msg='Remove the "cls_token" dimension')

                if self.patch_embed.num_patches != self.pos_embed.shape[1]:
                    # Upsample pos_embed weights for adapting training inputs.
                    h, w = self.img_size
                    pos_embed = VisionTransformer.upsample_pos_embed(
                        self.pos_embed, h, w, self.pretrain_img_size,
                        self.pretrain_img_size, self.patch_size)
                    self.pos_embed = nn.Parameter(pos_embed)

        elif pretrained is None:
            # We only implement the 'jax_impl' initialization implemented at
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353
            normal_init(self.pos_embed)
            for n, m in self.named_modules():
                if isinstance(m, Linear):
                    xavier_init(m.weight, distribution='uniform')
                    if m.bias is not None:
                        if 'mlp' in n:
                            normal_init(m.bias, std=1e-6)
                        else:
                            constant_init(m.bias, 0)
                elif isinstance(m, Conv2d):
                    kaiming_init(m.weight, mode='fan_in')
                    if m.bias is not None:
                        constant_init(m.bias, 0)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m.bias, 0)
                    constant_init(m.weight, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def _pos_embeding(self, img, patched_img, pos_embed):
        """Positiong embeding method.

        Resize the pos_embed, it input image size not equal to training
            image size.
        Args:
            img (torch.Tensor): The inference image tensor, the shape
                must be [B, C, H, W].
            patched_img (torch.Tensor): The patched image, it should be
                shape of [B, L1, C].
            pos_embed (torch.Tensor): The pos_embed weighs, it should be
                shape of [B, L2, c].
        Return:
            torch.Tensor: The pos encoded image feature.
        """
        assert len(patched_img.shape) == 3 and len(pos_embed.shape) == 3, \
            'the shapes of patched_img and pos_embed must be [B, L, C]'
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            # If run tools/test.py to test a model, the shape of pos_embed
            # should be [B, L_pretrain, C]
            if pos_len == (self.pretrain_img_size // self.patch_size)**2:
                pos_h, pos_w = self.pretrain_img_size, self.pretrain_img_size
            # During evaluation at training, the shape of pos_embed should be
            # [B, L_training, C]
            elif pos_len == (self.img_size[0] // self.patch_size) * (
                    self.img_size[1] // self.patch_size):
                pos_h, pos_w = self.img_size
            else:
                raise ValueError('Unexpected shape of pos_embed, got {}.',
                                 pos_embed.shape)
            input_h, input_w = img.shape[2:]
            pos_embed = VisionTransformer.upsample_pos_embed(
                pos_embed, input_h, input_w, pos_h, pos_w, self.patch_size)
        return patched_img + pos_embed

    @staticmethod
    def upsample_pos_embed(pos_embed, input_h, input_w, pos_h, pos_w,
                           patch_size):
        """Upsample pos_embed weights."""
        assert len(
            pos_embed.shape) == 3, 'shape of pos_embed must be [B, L, C]'
        pos_embed = pos_embed.reshape(1, pos_h // patch_size,
                                      pos_w // patch_size,
                                      pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(
            pos_embed,
            size=[input_h // patch_size, input_w // patch_size],
            align_corners=False,
            mode='bicubic')
        pos_embed = torch.flatten(pos_embed, 2).transpose(1, 2)
        return pos_embed

    def forward(self, inputs):
        x = self.patch_embed(inputs)
        x = self._pos_embeding(inputs, x, self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        B, _, C = x.shape
        x = x.reshape(B, inputs.shape[2] // self.patch_size,
                      inputs.shape[3] // self.patch_size,
                      C).permute(0, 3, 1, 2)
        return [x]

    def train(self, mode=True):
        super(VisionTransformer, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()
