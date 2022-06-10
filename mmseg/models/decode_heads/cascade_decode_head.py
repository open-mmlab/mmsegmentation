# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from .decode_head import BaseDecodeHead


class BaseCascadeDecodeHead(BaseDecodeHead, metaclass=ABCMeta):
    """Base class for cascade decode head used in
    :class:`CascadeEncoderDecoder."""

    def __init__(self, *args, **kwargs):
        super(BaseCascadeDecodeHead, self).__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, inputs, prev_output):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, prev_output, batch_data_samples,
                      train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs, prev_output)
        losses = self.losses(seg_logits, batch_data_samples)

        return losses

    def forward_test(self, inputs, prev_output, batch_img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs, prev_output)
