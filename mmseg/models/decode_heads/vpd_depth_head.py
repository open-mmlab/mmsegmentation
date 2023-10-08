# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmengine.model import BaseModule
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import SampleList
from ..builder import build_loss
from ..utils import resize
from .decode_head import BaseDecodeHead


class VPDDepthDecoder(BaseModule):
    """VPD Depth Decoder class.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_deconv_layers (int): Number of deconvolution layers.
        num_deconv_filters (List[int]): List of output channels for
            deconvolution layers.
        init_cfg (Optional[Union[Dict, List[Dict]]], optional): Configuration
            for weight initialization. Defaults to Normal for Conv2d and
            ConvTranspose2d layers.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_deconv_layers: int,
                 num_deconv_filters: List[int],
                 init_cfg: Optional[Union[Dict, List[Dict]]] = dict(
                     type='Normal',
                     std=0.001,
                     layer=['Conv2d', 'ConvTranspose2d'])):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels

        self.deconv_layers = self._make_deconv_layer(
            num_deconv_layers,
            num_deconv_filters,
        )

        conv_layers = []
        conv_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=num_deconv_filters[-1],
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1))
        conv_layers.append(build_norm_layer(dict(type='BN'), out_channels)[1])
        conv_layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*conv_layers)

        self.up_sample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        """Forward pass through the decoder network."""
        out = self.deconv_layers(x)
        out = self.conv_layers(out)

        out = self.up_sample(out)
        out = self.up_sample(out)

        return out

    def _make_deconv_layer(self, num_layers, num_deconv_filters):
        """Make deconv layers."""

        layers = []
        in_channels = self.in_channels
        for i in range(num_layers):

            num_channels = num_deconv_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=in_channels,
                    out_channels=num_channels,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    output_padding=0,
                    bias=False))
            layers.append(nn.BatchNorm2d(num_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = num_channels

        return nn.Sequential(*layers)


@MODELS.register_module()
class VPDDepthHead(BaseDecodeHead):
    """Depth Prediction Head for VPD.

    .. _`VPD`: https://arxiv.org/abs/2303.02153

    Args:
        max_depth (float): Maximum depth value. Defaults to 10.0.
        in_channels (Sequence[int]): Number of input channels for each
            convolutional layer.
        embed_dim (int): Dimension of embedding. Defaults to 192.
        feature_dim (int): Dimension of aggregated feature. Defaults to 1536.
        num_deconv_layers (int): Number of deconvolution layers in the
            decoder. Defaults to 3.
        num_deconv_filters (Sequence[int]): Number of filters for each deconv
            layer. Defaults to (32, 32, 32).
        fmap_border (Union[int, Sequence[int]]): Feature map border for
            cropping. Defaults to 0.
        align_corners (bool): Flag for align_corners in interpolation.
            Defaults to False.
        loss_decode (dict): Configurations for the loss function. Defaults to
            dict(type='SiLogLoss').
        init_cfg (dict): Initialization configurations. Defaults to
            dict(type='TruncNormal', std=0.02, layer=['Conv2d', 'Linear']).
    """

    num_classes = 1
    out_channels = 1
    input_transform = None

    def __init__(
        self,
        max_depth: float = 10.0,
        in_channels: Sequence[int] = [320, 640, 1280, 1280],
        embed_dim: int = 192,
        feature_dim: int = 1536,
        num_deconv_layers: int = 3,
        num_deconv_filters: Sequence[int] = (32, 32, 32),
        fmap_border: Union[int, Sequence[int]] = 0,
        align_corners: bool = False,
        loss_decode: dict = dict(type='SiLogLoss'),
        init_cfg=dict(
            type='TruncNormal', std=0.02, layer=['Conv2d', 'Linear']),
    ):

        super(BaseDecodeHead, self).__init__(init_cfg=init_cfg)

        # initialize parameters
        self.in_channels = in_channels
        self.max_depth = max_depth
        self.align_corners = align_corners

        # feature map border
        if isinstance(fmap_border, int):
            fmap_border = (fmap_border, fmap_border)
        self.fmap_border = fmap_border

        # define network layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0], 3, stride=2, padding=1),
            nn.GroupNorm(16, in_channels[0]),
            nn.ReLU(),
            nn.Conv2d(in_channels[0], in_channels[0], 3, stride=2, padding=1),
        )
        self.conv2 = nn.Conv2d(
            in_channels[1], in_channels[1], 3, stride=2, padding=1)

        self.conv_aggregation = nn.Sequential(
            nn.Conv2d(sum(in_channels), feature_dim, 1),
            nn.GroupNorm(16, feature_dim),
            nn.ReLU(),
        )

        self.decoder = VPDDepthDecoder(
            in_channels=embed_dim * 8,
            out_channels=embed_dim,
            num_deconv_layers=num_deconv_layers,
            num_deconv_filters=num_deconv_filters)

        self.depth_pred_layer = nn.Sequential(
            nn.Conv2d(
                embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(embed_dim, 1, kernel_size=3, stride=1, padding=1))

        # build loss
        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_depth_maps = [
            data_sample.gt_depth_map.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_depth_maps, dim=0)

    def forward(self, x):
        x = [
            x[0], x[1],
            torch.cat([x[2], F.interpolate(x[3], scale_factor=2)], dim=1)
        ]
        x = torch.cat([self.conv1(x[0]), self.conv2(x[1]), x[2]], dim=1)
        x = self.conv_aggregation(x)

        x = x[:, :, :x.size(2) - self.fmap_border[0], :x.size(3) -
              self.fmap_border[1]].contiguous()
        x = self.decoder(x)
        out = self.depth_pred_layer(x)

        depth = torch.sigmoid(out) * self.max_depth

        return depth

    def loss_by_feat(self, pred_depth_map: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute depth estimation loss.

        Args:
            pred_depth_map (Tensor): The output from decode head forward
                function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_dpeth_map`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        gt_depth_map = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        pred_depth_map = resize(
            input=pred_depth_map,
            size=gt_depth_map.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    pred_depth_map, gt_depth_map)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    pred_depth_map, gt_depth_map)

        return loss
