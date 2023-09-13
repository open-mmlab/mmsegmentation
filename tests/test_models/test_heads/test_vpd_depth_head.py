# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.structures import PixelData

from mmseg.models.decode_heads import VPDDepthHead
from mmseg.structures import SegDataSample


class TestVPDDepthHead(TestCase):

    def setUp(self):
        """Set up common resources."""
        self.in_channels = [320, 640, 1280, 1280]
        self.max_depth = 10.0
        self.loss_decode = dict(
            type='SiLogLoss'
        )  # Replace with your actual loss type and parameters
        self.vpd_depth_head = VPDDepthHead(
            max_depth=self.max_depth,
            in_channels=self.in_channels,
            loss_decode=self.loss_decode)

    def test_forward(self):
        """Test the forward method."""
        # Create a mock input tensor. Replace shape as per your needs.
        x = [
            torch.randn(1, 320, 32, 32),
            torch.randn(1, 640, 16, 16),
            torch.randn(1, 1280, 8, 8),
            torch.randn(1, 1280, 4, 4)
        ]

        output = self.vpd_depth_head.forward(x)
        print(output.shape)

        self.assertEqual(output.shape, (1, 1, 256, 256))

    def test_loss_by_feat(self):
        """Test the loss_by_feat method."""
        # Create mock data for `pred_depth_map` and `batch_data_samples`.
        pred_depth_map = torch.randn(1, 1, 32, 32)
        gt_depth_map = PixelData(data=torch.rand(1, 32, 32))
        batch_data_samples = [SegDataSample(gt_depth_map=gt_depth_map)]

        loss = self.vpd_depth_head.loss_by_feat(pred_depth_map,
                                                batch_data_samples)

        self.assertIsNotNone(loss)
