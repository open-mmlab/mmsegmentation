# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmengine.structures import PixelData
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import SampleList
from .fcn_head import FCNHead


@MODELS.register_module()
class STDCHead(FCNHead):
    """This head is the implementation of `Rethinking BiSeNet For Real-time
    Semantic Segmentation <https://arxiv.org/abs/2104.13188>`_.

    Args:
        boundary_threshold (float): The threshold of calculating boundary.
            Default: 0.1.
    """

    def __init__(self, boundary_threshold=0.1, **kwargs):
        super().__init__(**kwargs)
        self.boundary_threshold = boundary_threshold
        # Using register buffer to make laplacian kernel on the same
        # device of `seg_label`.
        self.register_buffer(
            'laplacian_kernel',
            torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1],
                         dtype=torch.float32,
                         requires_grad=False).reshape((1, 1, 3, 3)))
        self.fusion_kernel = torch.nn.Parameter(
            torch.tensor([[6. / 10], [3. / 10], [1. / 10]],
                         dtype=torch.float32).reshape(1, 3, 1, 1),
            requires_grad=False)

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute Detail Aggregation Loss."""
        # Note: The paper claims `fusion_kernel` is a trainable 1x1 conv
        # parameters. However, it is a constant in original repo and other
        # codebase because it would not be added into computation graph
        # after threshold operation.
        seg_label = self._stack_batch_gt(batch_data_samples).to(
            self.laplacian_kernel)
        boundary_targets = F.conv2d(
            seg_label, self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > self.boundary_threshold] = 1
        boundary_targets[boundary_targets <= self.boundary_threshold] = 0

        boundary_targets_x2 = F.conv2d(
            seg_label, self.laplacian_kernel, stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)

        boundary_targets_x4 = F.conv2d(
            seg_label, self.laplacian_kernel, stride=4, padding=1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

        boundary_targets_x4_up = F.interpolate(
            boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(
            boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')

        boundary_targets_x2_up[
            boundary_targets_x2_up > self.boundary_threshold] = 1
        boundary_targets_x2_up[
            boundary_targets_x2_up <= self.boundary_threshold] = 0

        boundary_targets_x4_up[
            boundary_targets_x4_up > self.boundary_threshold] = 1
        boundary_targets_x4_up[
            boundary_targets_x4_up <= self.boundary_threshold] = 0

        boundary_targets_pyramids = torch.stack(
            (boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up),
            dim=1)

        boundary_targets_pyramids = boundary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = F.conv2d(boundary_targets_pyramids,
                                           self.fusion_kernel)

        boudary_targets_pyramid[
            boudary_targets_pyramid > self.boundary_threshold] = 1
        boudary_targets_pyramid[
            boudary_targets_pyramid <= self.boundary_threshold] = 0

        seg_labels = boudary_targets_pyramid.long()
        batch_sample_list = []
        for label in seg_labels:
            seg_data_sample = SegDataSample()
            seg_data_sample.gt_sem_seg = PixelData(data=label)
            batch_sample_list.append(seg_data_sample)

        loss = super().loss_by_feat(seg_logits, batch_sample_list)
        return loss
