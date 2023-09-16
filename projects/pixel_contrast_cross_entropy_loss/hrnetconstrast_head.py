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
import warnings

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

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

    def __init__(self, in_channels: int, proj_n=256, proj_mode='convmlp'):
        super().__init__()
        if proj_mode == 'linear':
            self.proj = nn.Conv2d(in_channels, proj_n, kernel_size=1)
        elif proj_mode == 'convmlp':
            self.proj = nn.Sequential(
                ConvModule(in_channels, in_channels, kernel_size=1),
                nn.Conv2d(in_channels, proj_n, kernel_size=1),
            )
        else:
            raise KeyError("The type of project head only support 'linear' \
                        and 'convmlp', but got {}.".format(proj_mode))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.proj(x), p=2.0, dim=1)


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

    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 proj_n=256,
                 proj_mode='convmlp',
                 drop_p=0.1,
                 seg_head=dict(
                     type='FCNHead',
                     in_channels=[18, 36, 72, 144],
                     in_index=(0, 1, 2, 3),
                     channels=sum([18, 36, 72, 144]),
                     input_transform='resize_concat',
                     kernel_size=1,
                     num_convs=1,
                     concat_input=False,
                     dropout_ratio=-1,
                     num_classes=19,
                     norm_cfg=dict(type='SyncBN', requires_grad=True),
                     align_corners=False),
                 loss_decode=[
                    dict(type='PixelContrastCrossEntropyLoss',
                        base_temperature=0.07,
                        temperature=0.1,
                        ignore_index=255,
                        max_samples=1024,
                        max_views=100,
                        loss_weight=0.1),
                    dict(type='CrossEntropyLoss', loss_weight=1.0)
                    ],
                    **kwargs):
        super().__init__(in_channels,
                 channels,
                 num_classes = num_classes,
                 loss_decode = loss_decode,
                 init_cfg = dict(type='Normal', std=0.01),
                 **kwargs)


        if proj_n <= 0:
            raise KeyError('proj_n must >0')
        if drop_p < 0 or drop_p > 1 or not isinstance(drop_p, float):
            raise KeyError('drop_p must be a float >=0')
        self.proj_n = proj_n
            
        self.seghead = MODELS.build(seg_head)
        self.projhead = ProjectionHead(
            in_channels=self.in_channels, proj_n=proj_n, proj_mode=proj_mode)
        del self.conv_seg
        
    def cls_seg(self):
        '''Remove cls_seg, or distributed training will encounter an error'''
        pass

    def forward(self, inputs):
        output = []
        output.append(self.seghead(inputs))
        inputs = self._transform_inputs(inputs)
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
            if loss_decode.loss_name in ['loss_ce']:
                pred = F.interpolate(
                    input=seg_logits[0],
                    size=seg_label.shape[-2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
                if loss_decode.loss_name not in loss:
                    loss[loss_decode.loss_name] = loss_decode(
                        pred,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                else:
                    loss[loss_decode.loss_name] += loss_decode(
                        pred,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
            elif loss_decode.loss_name == 'loss_pixel_contrast_cross_entropy':
                if loss_decode.loss_name not in loss:
                    loss[loss_decode.loss_name] = loss_decode(
                        seg_logits, seg_label)
                else:
                    loss[loss_decode.loss_name] += loss_decode(
                        seg_logits, seg_label)
            else:
                raise KeyError('loss_name not matched')

        loss['acc_seg'] = accuracy(
            F.interpolate(seg_logits[0], 
                          size=seg_label.shape[-2:],
                          mode='bilinear',
                          align_corners=self.align_corners),
            seg_label,
            ignore_index=self.ignore_index)
        return loss

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """

        seg_logits = resize(
            input=seg_logits[0],
            size=batch_img_metas[0]['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits

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
