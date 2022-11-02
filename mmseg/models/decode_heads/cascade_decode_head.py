# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List

from torch import Tensor

from mmseg.utils import ConfigType
from .decode_head import BaseDecodeHead


class BaseCascadeDecodeHead(BaseDecodeHead, metaclass=ABCMeta):
    """Base class for cascade decode head used in
    :class:`CascadeEncoderDecoder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, inputs, prev_output):
        """Placeholder of forward function."""
        pass

    def loss(self, inputs: List[Tensor], prev_output: Tensor,
             batch_data_samples: List[dict], train_cfg: ConfigType) -> Tensor:
        """Forward function for training.

        Args:
            inputs (List[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs, prev_output)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)

        return losses

    def predict(self, inputs: List[Tensor], prev_output: Tensor,
                batch_img_metas: List[dict], tese_cfg: ConfigType):
        """Forward function for testing.

        Args:
            inputs (List[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        seg_logits = self.forward(inputs, prev_output)

        return self.predict_by_feat(seg_logits, batch_img_metas)
