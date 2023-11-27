# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from torch import Tensor

from mmseg.models.decode_heads.sep_aspp_head import DepthwiseSeparableASPPHead
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import SampleList


class ProjectionHead(nn.Module):
    """ProjectionHead, project feature map to specific channels.

    Args:
        dim_in (int): Input channels.
        norm_cfg (dict): config of norm layer.
        proj_dim (int): Output channels. Default: 256.
        proj (str): Projection type, 'linear' or 'convmlp'. Default: 'convmlp'
    """

    def __init__(self,
                 dim_in: int,
                 norm_cfg: dict,
                 proj_dim: int = 256,
                 proj: str = 'convmlp'):
        super().__init__()
        assert proj in ['convmlp', 'linear']
        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                build_norm_layer(norm_cfg, dim_in)[1], nn.ReLU(inplace=True),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1))

    def forward(self, x):
        return torch.nn.functional.normalize(self.proj(x), p=2, dim=1)


@MODELS.register_module()
class DepthwiseSeparableASPPContrastHead(DepthwiseSeparableASPPHead):
    """Deep Hierarchical Semantic Segmentation. This head is the implementation
    of `<https://arxiv.org/abs/2203.14335>`_.

    Based on Encoder-Decoder with Atrous Separable Convolution for
    Semantic Image Segmentation.
    `DeepLabV3+ <https://arxiv.org/abs/1802.02611>`_.

    Args:
        proj (str): The type of ProjectionHead, 'linear' or 'convmlp',
            default 'convmlp'
    """

    def __init__(self, proj: str = 'convmlp', **kwargs):
        super().__init__(**kwargs)
        self.proj_head = ProjectionHead(
            dim_in=2048, norm_cfg=self.norm_cfg, proj=proj)
        self.register_buffer('step', torch.zeros(1))

    def forward(self, inputs) -> Tuple[Tensor]:
        """Forward function."""
        output = super().forward(inputs)

        self.step += 1
        embedding = self.proj_head(inputs[-1])

        return output, embedding

    def predict_by_feat(self, seg_logits: Tuple[Tensor],
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        # HSSN decode_head output is: (out, embedding): tuple
        # only need 'out' here.
        if isinstance(seg_logits, tuple):
            seg_logit = seg_logits[0]

        if seg_logit.size(1) == 26:  # For cityscapes dataset，19 + 7
            hiera_num_classes = 7
            seg_logit[:, 0:2] += seg_logit[:, -7]
            seg_logit[:, 2:5] += seg_logit[:, -6]
            seg_logit[:, 5:8] += seg_logit[:, -5]
            seg_logit[:, 8:10] += seg_logit[:, -4]
            seg_logit[:, 10:11] += seg_logit[:, -3]
            seg_logit[:, 11:13] += seg_logit[:, -2]
            seg_logit[:, 13:19] += seg_logit[:, -1]

        elif seg_logit.size(1) == 12:  # For Pascal_person dataset, 7 + 5
            hiera_num_classes = 5
            seg_logit[:, 0:1] = seg_logit[:, 0:1] + \
                seg_logit[:, 7] + seg_logit[:, 10]
            seg_logit[:, 1:5] = seg_logit[:, 1:5] + \
                seg_logit[:, 8] + seg_logit[:, 11]
            seg_logit[:, 5:7] = seg_logit[:, 5:7] + \
                seg_logit[:, 9] + seg_logit[:, 11]

        elif seg_logit.size(1) == 25:  # For LIP dataset, 20 + 5
            hiera_num_classes = 5
            seg_logit[:, 0:1] = seg_logit[:, 0:1] + \
                seg_logit[:, 20] + seg_logit[:, 23]
            seg_logit[:, 1:8] = seg_logit[:, 1:8] + \
                seg_logit[:, 21] + seg_logit[:, 24]
            seg_logit[:, 10:12] = seg_logit[:, 10:12] + \
                seg_logit[:, 21] + seg_logit[:, 24]
            seg_logit[:, 13:16] = seg_logit[:, 13:16] + \
                seg_logit[:, 21] + seg_logit[:, 24]
            seg_logit[:, 8:10] = seg_logit[:, 8:10] + \
                seg_logit[:, 22] + seg_logit[:, 24]
            seg_logit[:, 12:13] = seg_logit[:, 12:13] + \
                seg_logit[:, 22] + seg_logit[:, 24]
            seg_logit[:, 16:20] = seg_logit[:, 16:20] + \
                seg_logit[:, 22] + seg_logit[:, 24]

        # elif seg_logit.size(1) == 144 # For Mapillary dataset, 124+16+4
        # unofficial repository not release mapillary until 2023/2/6

        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
            # slide inference
            size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']
        seg_logit = seg_logit[:, :-hiera_num_classes]
        seg_logit = resize(
            input=seg_logit,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)

        return seg_logit

    def loss_by_feat(
            self,
            seg_logits: Tuple[Tensor],  # (out, embedding)
            batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss. Will fix in future.

        Args:
            seg_logits (Tuple[Tensor]): The output from decode head
                forward function.
                For this decode_head output are (out, embedding): tuple
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logit_before = seg_logits[0]
        embedding = seg_logits[1]
        seg_label = self._stack_batch_gt(batch_data_samples)

        loss = dict()
        seg_logit = resize(
            input=seg_logit_before,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        seg_logit_before = resize(
            input=seg_logit_before,
            scale_factor=0.5,
            mode='bilinear',
            align_corners=self.align_corners)

        loss['loss_seg'] = self.loss_decode(
            self.step,
            embedding,
            seg_logit_before,
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss
