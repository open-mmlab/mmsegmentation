# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
"""Modified from https://github.com/PaddlePaddle/PaddleSeg/
blob/2c8c35a8949fef74599f5ec557d340a14415f20d/
paddleseg/models/hrnet_contrast.py(Apache-2.0 License)"""

from typing import List, Tuple

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch import Tensor

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy
from mmseg.registry import MODELS
from mmseg.utils import ConfigType, SampleList


class ProjectionHead(nn.Module):
    """The projection head used by contrast learning.

    Args:
        dim_in (int):
            The dimensions of input features.
        proj_dim (int, optional):
            The output dimensions of projection head. Default: 256.
        proj (str, optional): The type of projection head,
            only support 'linear' and 'convmlp'. Default: 'convmlp'.
    """

    def __init__(self, in_channels: int, out_channels=256, proj='convmlp'):
        super().__init__()
        if proj == 'linear':
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                ConvModule(in_channels, in_channels, kernel_size=1),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
        else:
            raise KeyError("The type of project head only support 'linear' \
                        and 'convmlp', but got {}.".format(proj))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.proj(x), p=2.0, dim=1)


class SegmentationHead(nn.Module):
    """The segmentation head used by contrast learning.

    Args:
        dim_in (int):
            The dimensions of input features.
        proj_dim (int, optional):
            The output dimensions of projection head. Default: 256.
        proj (str, optional):
            The type of projection head,
            only support 'linear' and 'convmlp'. Default: 'convmlp'.
    """

    def __init__(self, in_channels: int, out_channels=19, drop_prob=0.1):
        super().__init__()

        self.seg = nn.Sequential(
            ConvModule(in_channels, in_channels, kernel_size=1),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.seg(x)


@MODELS.register_module()
class ContrastHead(BaseDecodeHead):
    """The segmentation head used by contrast learning.

    Args:
        drop_p (float):
            The probability of dropout in segment head.
        proj_n (int):
            Each pixel will be projected into a vector with length of proj_n.
        proj_mode (str):
            The mode for project head ,'linear' or 'convmlp'.
    """

    def __init__(self, drop_p=0.1, proj_n=256, proj_mode='convmlp', **kwargs):
        super().__init__(**kwargs)
        if proj_n <= 0:
            raise KeyError('proj_n must >0')
        if drop_p < 0 or drop_p > 1 or not isinstance(drop_p, float):
            raise KeyError('drop_p must be a float >=0')
        self.proj_n = proj_n

        self.seghead = SegmentationHead(
            in_channels=self.in_channels,
            out_channels=self.num_classes,
            drop_prob=drop_p)
        self.projhead = ProjectionHead(
            in_channels=self.in_channels, out_channels=proj_n, proj=proj_mode)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        output = []
        output.append(self.seghead(inputs))
        output.append(self.projhead(inputs))

        return output

    def loss_by_feat(self, seg_logits: List,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (List): The output from decode head forward function.
                seg_logits[0] is the output of seghead
                seg_logits[1] is the output of projhead
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits[0], seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for loss_decode in losses_decode:
            if loss_decode.loss_name == 'loss_ce':
                pred = F.interpolate(
                    input=seg_logits[0],
                    size=seg_label.shape[-2:],
                    mode='bilinear',
                    align_corners=True)
                loss[loss_decode.loss_name] = loss_decode(
                    pred,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            elif loss_decode.loss_name == 'loss_pixel_contrast_cross_entropy':
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits, seg_label)
            else:
                raise KeyError('loss_name not matched')

        loss['acc_seg'] = accuracy(
            F.interpolate(seg_logits[0], seg_label.shape[1:]),
            seg_label,
            ignore_index=self.ignore_index)
        return loss

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        seg_logits = self.forward(inputs)

        return self.predict_by_feat(seg_logits, batch_img_metas)
