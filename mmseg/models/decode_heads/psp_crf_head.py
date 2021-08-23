import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPCRF.
    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


@HEADS.register_module()
class PSPCRFHead(BaseDecodeHead):
    """Pyramid Scene Parsing Network.
    This head is the implementation of
    `PSPNet <https://arxiv.org/abs/1612.01105>`_.
    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(PSPCRFHead, self).__init__(**kwargs)
        assert isinstance(pool_scales, (list, tuple))
        self.pool_scales = pool_scales
        self.psp_modules = PPM(
            self.pool_scales,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.crf = CRF(n_spatial_dims=2)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        output = self.cls_seg(output)
        output = self.crf(output)
        return output


class CRF(nn.Module):
    """
    Class for learning and inference in conditional random field model
    using mean field approximation
    and convolutional approximation in pairwise potentials term.
    Parameters
    ----------
    n_spatial_dims : int
        Number of spatial dimensions of input tensors.
    filter_size : int or sequence of ints
        Size of the gaussian filters in message passing.
        If it is a sequence its length must be equal to ``n_spatial_dims``.
    n_iter : int
        Number of iterations in mean field approximation.
    requires_grad : bool
        Whether or not to train CRF's parameters.
    returns : str
        Can be 'logits', 'proba', 'log-proba'.
    smoothness_weight : float
        Initial weight of smoothness kernel.
    smoothness_theta : float or sequence of floats
        Initial bandwidths for each spatial feature
        in the gaussian smoothness kernel.
        If it is a sequence its length must be equal to ``n_spatial_dims``.
    """

    def __init__(self, n_spatial_dims, filter_size=11, n_iter=5,
                 requires_grad=True, returns='logits', smoothness_weight=1,
                 smoothness_theta=1):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.n_iter = n_iter
        self.filter_size = np.broadcast_to(filter_size, n_spatial_dims)
        self.returns = returns
        self.requires_grad = requires_grad

        self._set_param('smoothness_weight', smoothness_weight)
        self._set_param('inv_smoothness_theta',
                        1 / np.broadcast_to(smoothness_theta, n_spatial_dims))

    def _set_param(self, name, init_value):
        setattr(self, name, nn.Parameter(torch.tensor(
            init_value, dtype=torch.float, requires_grad=self.requires_grad)))

    def forward(self, x, spatial_spacings=None, verbose=False):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)``
            with negative unary potentials,
            e.g. the CNN's output.
        spatial_spacings : array of floats or None
            Array of shape ``(batch_size, len(spatial))``
            with spatial spacings of tensors in batch ``x``.
            None is equivalent to all ones.
            Used to adapt spatial gaussian filters
            to different inputs' resolutions.
        verbose : bool
            Whether to display the iterations using tqdm-bar.
        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)``
            with logits or (log-)probabilities of assignment to each class.
        """
        batch_size, n_classes, *spatial = x.shape
        assert len(spatial) == self.n_spatial_dims

        # binary segmentation case
        if n_classes == 1:
            x = torch.cat([x, torch.zeros(x.shape).to(x)], dim=1)

        if spatial_spacings is None:
            spatial_spacings = np.ones((batch_size, self.n_spatial_dims))

        negative_unary = x.clone()

        for i in tqdm(range(self.n_iter), disable=not verbose):
            # normalizing
            x = F.softmax(x, dim=1)

            # message passing
            x = self.smoothness_weight * self._smoothing_filter(
                x, spatial_spacings)

            # compatibility transform
            x = self._compatibility_transform(x)

            # adding unary potentials
            x = negative_unary - x

        if self.returns == 'logits':
            ou = x
        elif self.returns == 'proba':
            ou = F.softmax(x, dim=1)
        elif self.returns == 'log-proba':
            ou = F.log_softmax(x, dim=1)
        else:
            raise ValueError(
                "Attribute returns must be logits, 'proba' or 'log-proba'.")

        if n_classes == 1:
            out = ou[:, 0] - ou[:, 1] if self.returns == 'logits' else ou[:, 0]
            out.unsqueeze_(1)

        return out

    def _smoothing_filter(self, x, spatial_spacings):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)``
            with negative unary potentials, e.g. logits.
        spatial_spacings : torch.tensor or None
            Tensor of shape ``(batch_size, len(spatial))``
            with spatial spacings of tensors in batch ``x``.
        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)``.
        """
        return torch.stack([
            self._single_smoothing_filter(
                x[i], spatial_spacings[i]) for i in range(x.shape[0])])

    @staticmethod
    def _pad(x, filter_size):
        padding = []
        for fs in filter_size:
            padding += 2 * [fs // 2]

        return F.pad(x, list(reversed(padding)))  # F.pad pads from the end

    def _single_smoothing_filter(self, x, spatial_spacing):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(n, *spatial)``.
        spatial_spacing : sequence of len(spatial) floats
        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(n, *spatial)``.
        """
        x = self._pad(x, self.filter_size)
        for i, dim in enumerate(range(1, x.ndim)):
            # reshape to (-1, 1, x.shape[dim])
            x = x.transpose(dim, -1)
            shape_before_flatten = x.shape[:-1]
            x = x.flatten(0, -2).unsqueeze(1)

            # 1d gaussian filtering
            kernel = self._create_gaussian_kernel1d(
                self.inv_smoothness_theta[i], spatial_spacing[i],
                self.filter_size[i]).view(1, 1, -1).to(x)
            x = F.conv1d(x, kernel)

            # reshape back to (n, *spatial)
            x = x.squeeze(1).view(*shape_before_flatten,
                                  x.shape[-1]).transpose(-1, dim)

        return x

    @staticmethod
    def _create_gaussian_kernel1d(inverse_theta, spacing, filter_size):
        """
        Parameters
        ----------
        inverse_theta : torch.tensor
            Tensor of shape ``(,)``
        spacing : float
        filter_size : int
        Returns
        -------
        kernel : torch.tensor
            Tensor of shape ``(filter_size,)``.
        """
        distances = spacing * torch.arange(
            -(filter_size // 2), filter_size // 2 + 1
        ).to(inverse_theta)
        kernel = torch.exp(-(distances * inverse_theta) ** 2 / 2)
        zero_center = torch.ones(filter_size).to(kernel)
        zero_center[filter_size // 2] = 0
        return kernel * zero_center

    def _compatibility_transform(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape ``(batch_size, n_classes, *spatial)``.
        Returns
        -------
        output : torch.tensor of shape ``(batch_size, n_classes, *spatial)``.
        """
        labels = torch.arange(x.shape[1])
        compatibility_matrix = self._compatibility_function(labels,
                                                            labels.unsqueeze(
                                                                1)).to(x)
        return torch.einsum('ij..., jk -> ik...', x, compatibility_matrix)

    @staticmethod
    def _compatibility_function(label1, label2):
        """
        Input tensors must be broadcastable.
        Parameters
        ----------
        label1 : torch.Tensor
        label2 : torch.Tensor
        Returns
        -------
        compatibility : torch.Tensor
        """
        return -(label1 == label2).float()
