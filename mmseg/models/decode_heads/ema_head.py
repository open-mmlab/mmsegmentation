import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead


def reduce_mean(tensor):
    """Reduce mean when distributed training."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


class EMAModule(nn.Module):
    """Expectation Maximization Attention Module used in EMANet.

    Args:
        channels (int): Channels of the whole module.
        num_bases (int): Number of bases.
        num_stages (int): Number of the EM iterations.
        epsilon (float): A small value for computation stability.
            Default: 1e-6.
    """

    def __init__(self,
                 channels,
                 num_bases,
                 num_stages,
                 momentum,
                 epsilon=1e-6):
        super(EMAModule, self).__init__()
        assert num_stages >= 1, 'num_stages must be at least 1!'
        self.num_bases = num_bases
        self.num_stages = num_stages
        self.momentum = momentum
        self.epsilon = epsilon

        bases = torch.zeros(1, channels, self.num_bases)
        bases.normal_(0, math.sqrt(2. / self.num_bases))
        # [1, num_classes, num_bases]
        bases = self._l2norm(bases, dim=1)
        self.register_buffer('bases', bases)

    def _l2norm(self, input, dim):
        """Normalize the inp tensor with l2-norm.

        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Args:
            input (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.
        """
        return input / (self.epsilon + input.norm(dim=dim, keepdim=True))

    def _l1norm(self, input, dim):
        """Normalize the inp tensor with l1-norm.

        Args:
            input (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.
        """
        return input / (self.epsilon + input.sum(dim=dim, keepdim=True))

    def forward(self, feats):
        """Forward function."""
        batch_size, channels, height, width = feats.size()
        # [batch_size, channels, height*width]
        feats = feats.view(batch_size, channels, height * width)
        # [batch_size, channels, num_bases]
        bases = self.bases.repeat(batch_size, 1, 1)

        with torch.no_grad():
            for i in range(self.num_stages):
                # [batch_size, height*width, num_bases]
                attention = torch.einsum('bcn,bck->bnk', feats, bases)
                attention = F.softmax(attention, dim=2)
                attention_normed = self._l1norm(attention, dim=1)
                # [batch_size, channels, num_bases]
                bases = torch.einsum('bcn,bnk->bck', feats, attention_normed)
                bases = self._l2norm(bases, dim=1)

        feats_recon = torch.einsum('bck,bnk->bcn', bases, attention)
        feats_recon = feats_recon.view(batch_size, channels, height, width)

        if self.training:
            bases = bases.mean(dim=0, keepdim=True)
            bases = reduce_mean(bases)
            bases = self._l2norm(bases, dim=1)
            self.bases = (1 -
                          self.momentum) * self.bases + self.momentum * bases

        return feats_recon


@HEADS.register_module()
class EMAHead(BaseDecodeHead):
    """Expectation Maximization Attention Networks for Semantic Segmentation.

    This head is the implementation of `EMANet
    <https://arxiv.org/abs/1907.13426>`_.

    Args:
        num_bases (int): Number of bases.
        num_stages (int): Number of the EM iterations.
        momentum (float): Momentum to update the base. Default: 0.1.
        epsilon (float): A small value for computation stability.
            Default: 1e-6.
    """

    def __init__(self,
                 ema_channels,
                 num_bases,
                 num_stages,
                 momentum=0.1,
                 epsilon=1e-6,
                 **kwargs):
        super(EMAHead, self).__init__(**kwargs)
        self.ema_channels = ema_channels
        self.num_bases = num_bases
        self.num_stages = num_stages
        self.momentum = momentum
        self.epsilon = epsilon
        self.ema_module = EMAModule(self.ema_channels, self.num_bases,
                                    self.num_stages, self.momentum,
                                    self.epsilon)

        self.ema_in_conv = ConvModule(
            self.in_channels,
            self.ema_channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # project (0, inf) -> (-inf, inf)
        self.ema_mid_conv = ConvModule(
            self.ema_channels,
            self.ema_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)
        for param in self.ema_mid_conv.parameters():
            param.requires_grad = False

        self.ema_out_conv = ConvModule(
            self.ema_channels,
            self.ema_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.bottleneck = ConvModule(
            self.ema_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        feats = self.ema_in_conv(x)
        identity = feats
        feats = self.ema_mid_conv(feats)
        recon = self.ema_module(feats)
        recon = F.relu(recon, inplace=True)
        recon = self.ema_out_conv(recon)
        output = F.relu(identity + recon, inplace=True)
        output = self.bottleneck(output)
        output = self.cls_seg(output)
        return output
