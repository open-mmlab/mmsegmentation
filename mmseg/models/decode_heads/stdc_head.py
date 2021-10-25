# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from ..builder import HEADS
from ..losses.cross_entropy_loss import (binary_cross_entropy, cross_entropy,
                                         mask_cross_entropy)
from .fcn_head import FCNHead


@HEADS.register_module()
class STDCHead(FCNHead):
    """This head is the implementation of `Rethinking BiSeNet For Real-time
    Semantic Segmentation <https://arxiv.org/abs/2104.13188>`_.

    Args:
        add_lateral (bool): Whether use lateral connection to fuse features.
            Default: False.
        loss_da_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss', use_sigmoid=True).
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 ignore_index=255,
                 **kwargs):
        super(STDCHead, self).__init__(**kwargs)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.ignore_index = ignore_index
        # Using register buffer to make laplacian kernel on the same
        # device of `seg_label`.
        self.register_buffer(
            'laplacian_kernel',
            torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1],
                         dtype=torch.float32).reshape((1, 1, 3, 3)))
        self.fuse_kernel = torch.nn.Parameter(
            torch.tensor([[6. / 10], [3. / 10], [1. / 10]],
                         dtype=torch.float32).reshape(1, 3, 1, 1))

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)

        return output

    def losses(self, seg_logit, seg_label):
        """Compute Detail Aggregation Loss."""
        seg_label = seg_label.type(torch.float32)
        boundary_targets = F.conv2d(
            seg_label, self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

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

        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0

        boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0

        boudary_targets_pyramids = torch.stack(
            (boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up),
            dim=1)

        boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids,
                                           self.fuse_kernel)

        boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
        boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0

        if seg_logit.shape[-1] != boundary_targets.shape[-1]:
            seg_logit = F.interpolate(
                seg_logit,
                boundary_targets.shape[2:],
                mode='bilinear',
                align_corners=True)
        loss = dict()
        loss.update(
            super(STDCHead,
                  self).losses(seg_logit,
                               boudary_targets_pyramid.type(torch.long)))
        return loss
