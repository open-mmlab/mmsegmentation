import math

import torch
from torch import Number
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead


class EMAModule(nn.Module):
    """Expectation Maximization Attention Module used in EMANet.

    Args:
        channels (int): Channels of the whole module.
        num_bases (int): Number of bases.
        num_stages (int): Number of the EM iterations.
        epsilon (float): A small value for computation stability
    """

    def __init__(self, channels, num_bases, num_stages, momentum, epsilon):
        super(EMAModule, self).__init__()
        
        assert num_stages >= 1, 'num_stages must be at least 1!'
        self.num_bases = num_bases
        self.num_stages = num_stages
        self.momentum = momentum
        self.epsilon = epsilon

        bases = torch.Tensor(1, channels, self.num_bases)
        bases.normal_(0, math.sqrt(2. / self.num_bases))
        # [1, num_classes, num_bases]
        bases = self._l2norm(bases, dim=1)
        self.register_buffer('bases', bases)

    def _l2norm(self, input, dim):
        """Normlize the inp tensor with l2-norm.

        Returns a tensor where each sub-tensor of input along the given dim is 
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Args:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.
        """
        return input / (self.epsilon + input.norm(dim=dim, keepdim=True))

    def _l1norm(self, input, dim):
        """Normlize the inp tensor with l1-norm.

        Args:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.
        """
        return input / (self.epsilon + input.sum(dim=dim, keepdim=True))

    def forward(self, feats):
        """Forward function."""
        batch_size, num_classes, height, width = feats.size()
        # [batch_size, num_classes, height*width]
        feats = feats.view(batch_size, num_classes, height * width)
        # [batch_size, num_classes, num_bases]
        bases = self.bases.repeat(batch_size, 1, 1)

        def _one_stage(feats, bases):
            # [batch_size, height*width, num_bases]
            attention = torch.einsum('bcn,bck->bnk', feats, bases)
            attention = F.softmax(attention, dim=2)
            attention_normed = self._l1norm(attention, dim=1)
            # [batch_size, num_classes, num_bases]
            bases = torch.einsum('bcn,bnk->bck', feats, attention_normed)
            bases = self._l2norm(bases, dim=1)
            return bases, attention

        with torch.no_grad():
            for i in range(self.num_stages):
                bases, attention = _one_stage(feats, bases)

        base = dist.all_reduce(bases)
        base = self._l2norm(base, dim=1)
        self.base = (1 - self.momentum) * self.base + self.momentum * base

        feats_recon = torch.einsum('bck,bnk->bcn', bases, attention)
        feats_recon = feats_recon.view(batch_size, num_classes, height, width)
        return feats_recon


@HEADS.register_module()
class EMAHead(BaseDecodeHead):
    """Expectation Maximization Attention Networks for Semantic Segmentation.

    This head is the implementation of `EMANet
    <https://arxiv.org/abs/1907.13426>`_.

    Args:
        num_bases (int): Number of bases.
        num_stages (int): Number of the EM iterations.
        epsilon (float): A small value for computation stability
    """

    def __init__(self, num_bases, num_stages, momentum, epsilon, **kwargs):
        super(EMAHead, self).__init__(**kwargs)
        self.num_bases = num_bases
        self.num_stages = num_stages
        self.epsilon = epsilon
        self.ema_module = EMAModule(
            self.channels,
            self.num_bases,
            self.num_stages,
            self.momentum,
            self.epsilon)

        self.ema_in_conv = ConvModule(
            self.in_channels,
            self.channels,
            3,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.ema_mid_conv = ConvModule(
            self.channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)
        self.ema_out_conv = ConvModule(
            self.channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

    def forward(self, inputs):
        """Forward function."""
        feats = self.ema_in_conv(inputs)
        identity = feats
        feats = self.ema_mid_conv(feats)
        recon = self.ema_module(feats)
        recon = F.relu(recon, inplace=True)
        recon = self.ema_out_conv(recon)
        output = F.relu(identity + recon, inplace=True)
        return output
