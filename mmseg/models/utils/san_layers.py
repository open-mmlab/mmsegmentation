# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/MendelXu/SAN/blob/main/san/model/attn_helper.py  # noqa: E501
# Copyright (c) 2023 MendelXu.
# Licensed under the MIT License

import warnings
from typing import Optional

import torch
from mmcv.cnn.bricks.transformer import BaseTransformerLayer
from torch import Tensor, nn
from torch.nn import functional as F


def cross_attn_with_self_bias(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
):
    """Forward function of multi-head attention. Modified from
    multi_head_attention_forward in
    https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py.

    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
            Default: `True`
            Note: `needs_weight` defaults to `True`, but should be set to `False`
            For best performance when attention weights are not needed.
            *Setting needs_weights to `True`
            leads to a significant performance degradation.*
        attn_mask: 2D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    """  # noqa: E501
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, \
        'embed_dim must be divisible by num_heads'
    scaling = float(head_dim)**-0.5

    if not use_separate_proj_weight:
        if (query is key or torch.equal(
                query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            raise NotImplementedError('self-attention is not implemented')

        elif key is value or torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function
            # with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
                q_k = None
                q_v = None
            else:
                # This is inline in_proj function with
                # in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)
                q_k, q_v = F.linear(query, _w, _b).chunk(2, dim=-1)
        else:
            # This is inline in_proj function with
            # in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with
            # in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)
            q_k = F.linear(query, _w, _b)
            # This is inline in_proj function with
            # in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
            q_v = F.linear(query, _w, _b)
    else:
        q_proj_weight_non_opt = \
            torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = \
            torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = \
            torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt,
                         in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt,
                         in_proj_bias[embed_dim:(embed_dim * 2)])
            v = F.linear(value, v_proj_weight_non_opt,
                         in_proj_bias[(embed_dim * 2):])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert (
            attn_mask.dtype == torch.float32
            or attn_mask.dtype == torch.float64
            or attn_mask.dtype == torch.float16
            or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool
        ), 'Only float, byte, and bool types are supported for ' \
           'attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn('Byte tensor for attn_mask in nn.MultiheadAttention '
                          'is deprecated. Use bool tensor instead.')
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError(
                    'The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [
                    bsz * num_heads,
                    query.size(0), key.size(0)
            ]:
                raise RuntimeError(
                    'The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError(
                "attn_mask's dimension {} is not supported".format(
                    attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            'Byte tensor for key_padding_mask in nn.MultiheadAttention '
            'is deprecated. Use bool tensor instead.')
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, 'bias cannot be added to static key.'
            assert static_v is None, 'bias cannot be added to static value.'
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        q_k = q_k.contiguous().view(tgt_len, bsz * num_heads,
                                    head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        q_v = q_v.contiguous().view(tgt_len, bsz * num_heads,
                                    head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat(
            [
                k,
                torch.zeros(
                    (k.size(0), 1) + k.size()[2:],
                    dtype=k.dtype,
                    device=k.device),
            ],
            dim=1,
        )
        v = torch.cat(
            [
                v,
                torch.zeros(
                    (v.size(0), 1) + v.size()[2:],
                    dtype=v.dtype,
                    device=v.device),
            ],
            dim=1,
        )
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(
        attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len,
                                                       src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads,
                                                       tgt_len, src_len)
    # attn_out_weights: [bsz * num_heads, tgt_len, src_len]
    # ->[bsz * num_heads, tgt_len, src_len+1]
    self_weight = (q * q_k).sum(
        dim=-1, keepdim=True)  # [bsz * num_heads, tgt_len, 1]
    total_attn_output_weights = torch.cat([attn_output_weights, self_weight],
                                          dim=-1)
    total_attn_output_weights = F.softmax(total_attn_output_weights, dim=-1)
    total_attn_output_weights = F.dropout(
        total_attn_output_weights, p=dropout_p, training=training)
    attn_output_weights = \
        total_attn_output_weights[:, :, : -1]
    # [bsz * num_heads, tgt_len, src_len]
    self_weight = \
        total_attn_output_weights[:, :, -1:]  # [bsz * num_heads, tgt_len, 1]

    attn_output = torch.bmm(attn_output_weights,
                            v)  # [bsz * num_heads, tgt_len, head_dim]
    attn_output = (attn_output + self_weight * q_v
                   )  # [bsz * num_heads, tgt_len, head_dim]
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(
        tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len,
                                                       src_len)
        return attn_output, attn_output_weights  # .sum(dim=1) / num_heads
    else:
        return attn_output, None


def cross_attn_layer(tf_layer: BaseTransformerLayer, x, mem, attn_bias):
    """Implementation of transformer layer with cross attention. The cross
    attention shares the embedding weights with self-attention of tf_layer.
    Args:
        tf_layer: (TransformerEncoderLayer): The Module of transformer layer.
        x (Tensor): query [K,N,C]
        mem (Tensor): key and value [L,N,C]
        attn_bias (Tensor): attention bias [N*num_head,K,L]

    Return:
        x (Tensor): cross attention output [K,N,C]
    """
    self_attn_layer = tf_layer.attentions[0].attn
    attn_layer_paras = {
        'embed_dim_to_check': self_attn_layer.embed_dim,
        'num_heads': self_attn_layer.num_heads,
        'in_proj_weight': self_attn_layer.in_proj_weight,
        'in_proj_bias': self_attn_layer.in_proj_bias,
        'bias_k': self_attn_layer.bias_k,
        'bias_v': self_attn_layer.bias_v,
        'add_zero_attn': self_attn_layer.add_zero_attn,
        'dropout_p': self_attn_layer.dropout,
        'out_proj_weight': self_attn_layer.out_proj.weight,
        'out_proj_bias': self_attn_layer.out_proj.bias,
        'training': self_attn_layer.training
    }

    q_x = tf_layer.norms[0](x)
    k_x = v_x = tf_layer.norms[0](mem)
    x = x + cross_attn_with_self_bias(
        q_x,
        k_x,
        v_x,
        attn_mask=attn_bias,
        need_weights=False,
        **attn_layer_paras)[0]
    x = tf_layer.ffns[0](tf_layer.norms[1](x), identity=x)
    return x


class LayerNorm2d(nn.Module):
    """A LayerNorm variant, popularized by Transformers, that performs point-
    wise mean and variance normalization over the channel dimension for inputs
    that have shape (batch_size, channels, height, width).

    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x: torch.Tensor):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 affine_func=nn.Linear):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            affine_func(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
