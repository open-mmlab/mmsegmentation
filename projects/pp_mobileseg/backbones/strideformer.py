# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer
from mmcv.cnn.bricks.transformer import build_dropout
from mmengine.logging import print_log
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict

from mmseg.registry import MODELS


@MODELS.register_module()
class StrideFormer(BaseModule):
    """The StrideFormer implementation based on torch.

    The original article refers to:https://arxiv.org/abs/2304.05152
    Args:
        mobileV3_cfg(list): Each sublist describe the config for a
            MobileNetV3 block.
        channels(list): The input channels for each MobileNetV3 block.
        embed_dims(list): The channels of the features input to the sea
            attention block.
        key_dims(list, optional): The embeding dims for each head in
            attention.
        depths(list, optional): describes the depth of the attention block.
            i,e: M,N.
        num_heads(int, optional): The number of heads of the attention
            blocks.
        attn_ratios(int, optional): The expand ratio of V.
        mlp_ratios(list, optional): The ratio of mlp blocks.
        drop_path_rate(float, optional): The drop path rate in attention
            block.
        act_cfg(dict, optional): The activation layer of AAM:
            Aggregate Attention Module.
        inj_type(string, optional): The type of injection/AAM.
        out_channels(int, optional): The output channels of the AAM.
        dims(list, optional): The dimension of the fusion block.
        out_feat_chs(list, optional): The input channels of the AAM.
        stride_attention(bool, optional): whether to stride attention in
            each attention layer.
        pretrained(str, optional): the path of pretrained model.
    """

    def __init__(
        self,
        mobileV3_cfg,
        channels,
        embed_dims,
        key_dims=[16, 24],
        depths=[2, 2],
        num_heads=8,
        attn_ratios=2,
        mlp_ratios=[2, 4],
        drop_path_rate=0.1,
        act_cfg=dict(type='ReLU'),
        inj_type='AAM',
        out_channels=256,
        dims=(128, 160),
        out_feat_chs=None,
        stride_attention=True,
        pretrained=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        assert not (init_cfg and pretrained
                    ), 'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.depths = depths
        self.cfgs = mobileV3_cfg
        self.dims = dims
        for i in range(len(self.cfgs)):
            smb = StackedMV3Block(
                cfgs=self.cfgs[i],
                stem=True if i == 0 else False,
                in_channels=channels[i],
            )
            setattr(self, f'smb{i + 1}', smb)
        for i in range(len(depths)):
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depths[i])
            ]
            trans = BasicLayer(
                block_num=depths[i],
                embedding_dim=embed_dims[i],
                key_dim=key_dims[i],
                num_heads=num_heads,
                mlp_ratio=mlp_ratios[i],
                attn_ratio=attn_ratios,
                drop=0,
                attn_drop=0.0,
                drop_path=dpr,
                act_cfg=act_cfg,
                stride_attention=stride_attention,
            )
            setattr(self, f'trans{i + 1}', trans)

        self.inj_type = inj_type
        if self.inj_type == 'AAM':
            self.inj_module = InjectionMultiSumallmultiallsum(
                in_channels=out_feat_chs, out_channels=out_channels)
            self.feat_channels = [
                out_channels,
            ]
        elif self.inj_type == 'AAMSx8':
            self.inj_module = InjectionMultiSumallmultiallsumSimpx8(
                in_channels=out_feat_chs, out_channels=out_channels)
            self.feat_channels = [
                out_channels,
            ]
        elif self.inj_type == 'origin':
            for i in range(len(dims)):
                fuse = FusionBlock(
                    out_feat_chs[0] if i == 0 else dims[i - 1],
                    out_feat_chs[i + 1],
                    embed_dim=dims[i],
                    act_cfg=None,
                )
                setattr(self, f'fuse{i + 1}', fuse)
            self.feat_channels = [
                dims[i],
            ]
        else:
            raise NotImplementedError(self.inj_module + ' is not implemented')

        self.pretrained = pretrained
        # self.init_weights()

    def init_weights(self):
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type') == 'Pretrained'):
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            if 'pos_embed' in state_dict.keys():
                if self.pos_embed.shape != state_dict['pos_embed'].shape:
                    print_log(msg=f'Resize the pos_embed shape from '
                              f'{state_dict["pos_embed"].shape} to '
                              f'{self.pos_embed.shape}')
                    h, w = self.img_size
                    pos_size = int(
                        math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                    state_dict['pos_embed'] = self.resize_pos_embed(
                        state_dict['pos_embed'],
                        (h // self.patch_size, w // self.patch_size),
                        (pos_size, pos_size),
                        self.interpolate_mode,
                    )

            load_state_dict(self, state_dict, strict=False, logger=None)

    def forward(self, x):
        x_hw = x.shape[2:]
        outputs = []
        num_smb_stage = len(self.cfgs)
        num_trans_stage = len(self.depths)

        for i in range(num_smb_stage):
            smb = getattr(self, f'smb{i + 1}')
            x = smb(x)

            # 1/8 shared feat
            if i == 1:
                outputs.append(x)
            if num_trans_stage + i >= num_smb_stage:
                trans = getattr(
                    self, f'trans{i + num_trans_stage - num_smb_stage + 1}')
                x = trans(x)
                outputs.append(x)
        if self.inj_type == 'origin':
            x_detail = outputs[0]
            for i in range(len(self.dims)):
                fuse = getattr(self, f'fuse{i + 1}')

                x_detail = fuse(x_detail, outputs[i + 1])
            output = x_detail
        else:
            output = self.inj_module(outputs)

        return [output, x_hw]


class StackedMV3Block(nn.Module):
    """The MobileNetV3 block.

    Args:
        cfgs (list): The MobileNetV3 config list of a stage.
        stem (bool): Whether is the first stage or not.
        in_channels (int, optional): The channels of input image. Default: 3.
        scale: float=1.0.
        The coefficient that controls the size of network parameters.

    Returns:
        model: nn.Module.
        A stage of specific MobileNetV3 model depends on args.
    """

    def __init__(self,
                 cfgs,
                 stem,
                 in_channels,
                 scale=1.0,
                 norm_cfg=dict(type='BN')):
        super().__init__()

        self.scale = scale
        self.stem = stem

        if self.stem:
            self.conv = ConvModule(
                in_channels=3,
                out_channels=_make_divisible(in_channels * self.scale),
                kernel_size=3,
                stride=2,
                padding=1,
                groups=1,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='HSwish'),
            )

        self.blocks = nn.ModuleList()
        for i, (k, exp, c, se, act, s) in enumerate(cfgs):
            self.blocks.append(
                ResidualUnit(
                    in_channel=_make_divisible(in_channels * self.scale),
                    mid_channel=_make_divisible(self.scale * exp),
                    out_channel=_make_divisible(self.scale * c),
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    act=act,
                    dilation=1,
                ))
            in_channels = _make_divisible(self.scale * c)

    def forward(self, x):
        if self.stem:
            x = self.conv(x)
        for i, block in enumerate(self.blocks):
            x = block(x)

        return x


class ResidualUnit(nn.Module):
    """The Residual module.

    Args:
        in_channel (int, optional): The channels of input feature.
        mid_channel (int, optional): The channels of middle process.
        out_channel (int, optional): The channels of output feature.
        kernel_size (int, optional): The size of the convolving kernel.
        stride (int, optional): The stride size.
        use_se (bool, optional): if to use the SEModule.
        act (string, optional): activation layer.
        dilation (int, optional): The dilation size.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', requires_grad=True).
    """

    def __init__(
            self,
            in_channel,
            mid_channel,
            out_channel,
            kernel_size,
            stride,
            use_se,
            act=None,
            dilation=1,
            norm_cfg=dict(type='BN'),
    ):
        super().__init__()
        self.if_shortcut = stride == 1 and in_channel == out_channel
        self.if_se = use_se
        self.expand_conv = ConvModule(
            in_channels=in_channel,
            out_channels=mid_channel,
            kernel_size=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=dict(type=act) if act is not None else None,
        )
        self.bottleneck_conv = ConvModule(
            in_channels=mid_channel,
            out_channels=mid_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=int((kernel_size - 1) // 2) * dilation,
            bias=False,
            groups=mid_channel,
            dilation=dilation,
            norm_cfg=norm_cfg,
            act_cfg=dict(type=act) if act is not None else None,
        )
        if self.if_se:
            self.mid_se = SEModule(mid_channel)
        self.linear_conv = ConvModule(
            in_channels=mid_channel,
            out_channels=out_channel,
            kernel_size=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

    def forward(self, x):
        identity = x
        x = self.expand_conv(x)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = torch.add(identity, x)
        return x


class SEModule(nn.Module):
    """SE Module.

    Args:
        channel (int, optional): The channels of input feature.
        reduction (int, optional): The channel reduction rate.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='PReLU').
    """

    def __init__(self, channel, reduction=4, act_cfg=dict(type='ReLU')):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_act1 = ConvModule(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            norm_cfg=None,
            act_cfg=act_cfg,
        )

        self.conv_act2 = ConvModule(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            norm_cfg=None,
            act_cfg=dict(type='Hardsigmoid', slope=0.2, offset=0.5),
        )

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv_act1(x)
        x = self.conv_act2(x)
        return torch.mul(identity, x)


class BasicLayer(nn.Module):
    """The transformer basic layer.

    Args:
        block_num (int): the block nums of the transformer basic layer.
        embedding_dim (int): The feature dimension.
        key_dim (int): the key dim.
        num_heads (int): Parallel attention heads.
        mlp_ratio (float): the mlp ratio.
        attn_ratio (float): the attention ratio.
        drop (float): Probability of an element to be zeroed
            after the feed forward layer.Default: 0.0.
        attn_drop (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path (float): stochastic depth rate. Default 0.0.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='PReLU').
        stride_attention (bool, optional): whether to stride attention in
            each attention layer.
    """

    def __init__(
        self,
        block_num,
        embedding_dim,
        key_dim,
        num_heads,
        mlp_ratio=4.0,
        attn_ratio=2.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=None,
        act_cfg=None,
        stride_attention=None,
    ):
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(
                Block(
                    embedding_dim,
                    key_dim=key_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_ratio=attn_ratio,
                    drop=drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list) else drop_path,
                    act_cfg=act_cfg,
                    stride_attention=stride_attention,
                ))

    def forward(self, x):
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x


class Block(nn.Module):
    """the block of the transformer basic layer.

    Args:
        dim (int): The feature dimension.
        key_dim (int): The key dimension.
        num_heads (int): Parallel attention heads.
        mlp_ratio (float): the mlp ratio.
        attn_ratio (float): the attention ratio.
        drop (float): Probability of an element to be zeroed
            after the feed forward layer.Default: 0.0.
        drop_path (float): stochastic depth rate. Default 0.0.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='PReLU').
        stride_attention (bool, optional): whether to stride attention in
            each attention layer.
    """

    def __init__(
        self,
        dim,
        key_dim,
        num_heads,
        mlp_ratio=4.0,
        attn_ratio=2.0,
        drop=0.0,
        drop_path=0.0,
        act_cfg=None,
        stride_attention=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn = SeaAttention(
            dim,
            key_dim=key_dim,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            act_cfg=act_cfg,
            stride_attention=stride_attention,
        )
        self.drop_path = (
            build_dropout(dict(type='DropPath', drop_prob=drop_path))
            if drop_path > 0.0 else nn.Identity())
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_cfg=act_cfg,
            drop=drop,
        )

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))

        return x1


class SqueezeAxialPositionalEmbedding(nn.Module):
    """the Squeeze Axial Positional Embedding.

    Args:
        dim (int): The feature dimension.
        shape (int): The patch size.
    """

    def __init__(self, dim, shape):
        super().__init__()
        self.pos_embed = nn.init.normal_(
            nn.Parameter(torch.zeros(1, dim, shape)))

    def forward(self, x):
        B, C, N = x.shape
        x = x + F.interpolate(
            self.pos_embed, size=(N, ), mode='linear', align_corners=False)
        return x


class SeaAttention(nn.Module):
    """The sea attention.

    Args:
        dim (int): The feature dimension.
        key_dim (int):  The key dimension.
        num_heads (int): number of attention heads.
        attn_ratio (float): the attention ratio.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='PReLU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        stride_attention (bool, optional): whether to stride attention in
            each attention layer.
    """

    def __init__(
            self,
            dim,
            key_dim,
            num_heads,
            attn_ratio=4.0,
            act_cfg=None,
            norm_cfg=dict(type='BN'),
            stride_attention=False,
    ):

        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = ConvModule(
            dim, nh_kd, 1, bias=False, norm_cfg=norm_cfg, act_cfg=None)
        self.to_k = ConvModule(
            dim, nh_kd, 1, bias=False, norm_cfg=norm_cfg, act_cfg=None)

        self.to_v = ConvModule(
            dim, self.dh, 1, bias=False, norm_cfg=norm_cfg, act_cfg=None)
        self.stride_attention = stride_attention
        if self.stride_attention:
            self.stride_conv = ConvModule(
                dim,
                dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
                groups=dim,
                norm_cfg=norm_cfg,
                act_cfg=None,
            )

        self.proj = ConvModule(
            self.dh,
            dim,
            1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            order=('act', 'conv', 'norm'),
        )
        self.proj_encode_row = ConvModule(
            self.dh,
            self.dh,
            1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            order=('act', 'conv', 'norm'),
        )
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.proj_encode_column = ConvModule(
            self.dh,
            self.dh,
            1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            order=('act', 'conv', 'norm'),
        )
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.dwconv = ConvModule(
            2 * self.dh,
            2 * self.dh,
            3,
            padding=1,
            groups=2 * self.dh,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.pwconv = ConvModule(
            2 * self.dh, dim, 1, bias=False, norm_cfg=norm_cfg, act_cfg=None)
        self.sigmoid = build_activation_layer(dict(type='HSigmoid'))

    def forward(self, x):
        B, C, H_ori, W_ori = x.shape
        if self.stride_attention:
            x = self.stride_conv(x)
        B, C, H, W = x.shape

        q = self.to_q(x)  # [B, nhead*dim, H, W]
        k = self.to_k(x)
        v = self.to_v(x)

        qkv = torch.cat([q, k, v], dim=1)
        qkv = self.dwconv(qkv)
        qkv = self.pwconv(qkv)

        qrow = (self.pos_emb_rowq(q.mean(-1)).reshape(
            [B, self.num_heads, -1, H]).permute(
                (0, 1, 3, 2)))  # [B, nhead, H, dim]
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(
            [B, self.num_heads, -1, H])  # [B, nhead, dim, H]
        vrow = (v.mean(-1).reshape([B, self.num_heads, -1,
                                    H]).permute([0, 1, 3, 2])
                )  # [B, nhead, H, dim*attn_ratio]

        attn_row = torch.matmul(qrow, krow) * self.scale  # [B, nhead, H, H]
        attn_row = nn.functional.softmax(attn_row, dim=-1)

        xx_row = torch.matmul(attn_row, vrow)  # [B, nhead, H, dim*attn_ratio]
        xx_row = self.proj_encode_row(
            xx_row.permute([0, 1, 3, 2]).reshape([B, self.dh, H, 1]))

        # squeeze column
        qcolumn = (
            self.pos_emb_columnq(q.mean(-2)).reshape(
                [B, self.num_heads, -1, W]).permute([0, 1, 3, 2]))
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(
            [B, self.num_heads, -1, W])
        vcolumn = (
            torch.mean(v, -2).reshape([B, self.num_heads, -1,
                                       W]).permute([0, 1, 3, 2]))

        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = nn.functional.softmax(attn_column, dim=-1)

        xx_column = torch.matmul(attn_column, vcolumn)  # B nH W C
        xx_column = self.proj_encode_column(
            xx_column.permute([0, 1, 3, 2]).reshape([B, self.dh, 1, W]))

        xx = torch.add(xx_row, xx_column)  # [B, self.dh, H, W]
        xx = torch.add(v, xx)

        xx = self.proj(xx)
        xx = self.sigmoid(xx) * qkv
        if self.stride_attention:
            xx = F.interpolate(xx, size=(H_ori, W_ori), mode='bilinear')

        return xx


class MLP(nn.Module):
    """the Multilayer Perceptron.

    Args:
        in_features (int): the input feature.
        hidden_features (int): the hidden feature.
        out_features (int): the output feature.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='PReLU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        drop (float): Probability of an element to be zeroed.
            Default 0.0
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_cfg=None,
            norm_cfg=dict(type='BN'),
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ConvModule(
            in_features,
            hidden_features,
            kernel_size=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )
        self.dwconv = ConvModule(
            hidden_features,
            hidden_features,
            kernel_size=3,
            padding=1,
            groups=hidden_features,
            norm_cfg=None,
            act_cfg=act_cfg,
        )

        self.fc2 = ConvModule(
            hidden_features,
            out_features,
            1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )
        self.drop = build_dropout(dict(type='Dropout', drop_prob=drop))

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FusionBlock(nn.Module):
    """The feature fusion block.

    Args:
        in_channel (int): the input channel.
        out_channel (int): the output channel.
        embed_dim (int): embedding dimension.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
    """

    def __init__(
            self,
            in_channel,
            out_channel,
            embed_dim,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
    ) -> None:
        super().__init__()
        self.local_embedding = ConvModule(
            in_channels=in_channel,
            out_channels=embed_dim,
            kernel_size=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

        self.global_act = ConvModule(
            in_channels=out_channel,
            out_channels=embed_dim,
            kernel_size=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg if act_cfg is not None else None,
        )

    def forward(self, x_l, x_g):
        """
        x_g: global features
        x_l: local features
        """
        B, C, H, W = x_l.shape

        local_feat = self.local_embedding(x_l)
        global_act = self.global_act(x_g)
        sig_act = F.interpolate(
            global_act, size=(H, W), mode='bilinear', align_corners=False)

        out = local_feat * sig_act

        return out


class InjectionMultiSumallmultiallsum(nn.Module):
    """the Aggregate Attention Module.

    Args:
        in_channels (tuple): the input channel.
        out_channels (int): the output channel.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
    """

    def __init__(
            self,
            in_channels=(64, 128, 256, 384),
            out_channels=256,
            act_cfg=dict(type='Sigmoid'),
            norm_cfg=dict(type='BN'),
    ):
        super().__init__()
        self.embedding_list = nn.ModuleList()
        self.act_embedding_list = nn.ModuleList()
        self.act_list = nn.ModuleList()
        for i in range(len(in_channels)):
            self.embedding_list.append(
                ConvModule(
                    in_channels=in_channels[i],
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False,
                    norm_cfg=norm_cfg,
                    act_cfg=None,
                ))
            self.act_embedding_list.append(
                ConvModule(
                    in_channels=in_channels[i],
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ))

    def forward(self, inputs):  # x_x8, x_x16, x_x32, x_x64
        low_feat1 = F.interpolate(inputs[0], scale_factor=0.5, mode='bilinear')
        low_feat1_act = self.act_embedding_list[0](low_feat1)
        low_feat1 = self.embedding_list[0](low_feat1)

        low_feat2 = F.interpolate(
            inputs[1], size=low_feat1.shape[-2:], mode='bilinear')
        low_feat2_act = self.act_embedding_list[1](low_feat2)  # x16
        low_feat2 = self.embedding_list[1](low_feat2)

        high_feat_act = F.interpolate(
            self.act_embedding_list[2](inputs[2]),
            size=low_feat2.shape[2:],
            mode='bilinear',
        )
        high_feat = F.interpolate(
            self.embedding_list[2](inputs[2]),
            size=low_feat2.shape[2:],
            mode='bilinear')

        res = (
            low_feat1_act * low_feat2_act * high_feat_act *
            (low_feat1 + low_feat2) + high_feat)

        return res


class InjectionMultiSumallmultiallsumSimpx8(nn.Module):
    """the Aggregate Attention Module.

    Args:
        in_channels (tuple): the input channel.
        out_channels (int): the output channel.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
    """

    def __init__(
            self,
            in_channels=(64, 128, 256, 384),
            out_channels=256,
            act_cfg=dict(type='Sigmoid'),
            norm_cfg=dict(type='BN'),
    ):
        super().__init__()
        self.embedding_list = nn.ModuleList()
        self.act_embedding_list = nn.ModuleList()
        self.act_list = nn.ModuleList()
        for i in range(len(in_channels)):
            if i != 1:
                self.embedding_list.append(
                    ConvModule(
                        in_channels=in_channels[i],
                        out_channels=out_channels,
                        kernel_size=1,
                        bias=False,
                        norm_cfg=norm_cfg,
                        act_cfg=None,
                    ))
            if i != 0:
                self.act_embedding_list.append(
                    ConvModule(
                        in_channels=in_channels[i],
                        out_channels=out_channels,
                        kernel_size=1,
                        bias=False,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                    ))

    def forward(self, inputs):
        # x_x8, x_x16, x_x32
        low_feat1 = self.embedding_list[0](inputs[0])

        low_feat2 = F.interpolate(
            inputs[1], size=low_feat1.shape[-2:], mode='bilinear')
        low_feat2_act = self.act_embedding_list[0](low_feat2)

        high_feat_act = F.interpolate(
            self.act_embedding_list[1](inputs[2]),
            size=low_feat2.shape[2:],
            mode='bilinear',
        )
        high_feat = F.interpolate(
            self.embedding_list[1](inputs[2]),
            size=low_feat2.shape[2:],
            mode='bilinear')

        res = low_feat2_act * high_feat_act * low_feat1 + high_feat

        return res


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@MODELS.register_module()
class Hardsigmoid(nn.Module):
    """the hardsigmoid activation.

    Args:
        slope (float, optional): The slope of hardsigmoid function.
            Default is 0.1666667.
        offset (float, optional): The offset of hardsigmoid function.
            Default is 0.5.
        inplace (bool): can optionally do the operation in-place.
            Default: ``False``
    """

    def __init__(self, slope=0.1666667, offset=0.5, inplace=False):
        super().__init__()
        self.slope = slope
        self.offset = offset

    def forward(self, x):
        return (x * self.slope + self.offset).clamp(0, 1)
