# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn
from torch.nn import functional as F


class LinearAttention(nn.Module):
    """Multi-Head linear attention proposed in "Transformers are RNNs".

    Source: https://github.com/KU-CVLAB/CAT-Seg/blob/main/cat_seg/modeling/transformer/model.py#L247 # noqa
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, queries, keys, values):
        """
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = F.elu(queries) + 1
        K = F.elu(keys) + 1

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum('nshd,nshv->nhdv', K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum('nlhd,nhd->nlh', Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum('nlhd,nhdv,nlh->nlhv', Q, KV,
                                      Z) * v_length

        return queried_values.contiguous()


class FullAttention(nn.Module):
    """Multi-head scaled dot-product attention, a.k.a full attention.

    Source: https://github.com/KU-CVLAB/CAT-Seg/blob/main/cat_seg/modeling/transformer/model.py#L276 # noqa
    """

    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum('nlhd,nshd->nlsh', queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(
                ~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]),
                float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum('nlsh,nshd->nlhd', A, values)

        return queried_values.contiguous()
