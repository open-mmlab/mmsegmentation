# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmseg.models.losses import SiLogLoss


class TestSiLogLoss(TestCase):

    def test_SiLogLoss_forward(self):
        pred = torch.tensor([[1.0, 2.0], [3.5, 4.0]], dtype=torch.float32)
        target = torch.tensor([[0.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        weight = torch.tensor([1.0, 0.5], dtype=torch.float32)

        loss_module = SiLogLoss()
        loss = loss_module.forward(pred, target, weight)

        expected_loss = 0.02
        self.assertAlmostEqual(loss.item(), expected_loss, places=2)
