# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.structures import InstanceData

from mmseg.models.assigners import HungarianAssigner


class TestHungarianAssigner(TestCase):

    def test_init(self):
        with self.assertRaises(AssertionError):
            HungarianAssigner([])

    def test_hungarian_match_assigner(self):
        assigner = HungarianAssigner([
            dict(type='ClassificationCost', weight=2.0),
            dict(type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
            dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0)
        ])
        num_classes = 3
        num_masks = 10
        num_points = 20
        gt_instances = InstanceData()
        gt_instances.labels = torch.randint(0, num_classes, (num_classes, ))
        gt_instances.masks = torch.randint(0, 2, (num_classes, num_points))
        pred_instances = InstanceData()
        pred_instances.scores = torch.rand((num_masks, num_classes))
        pred_instances.masks = torch.rand((num_masks, num_points))

        matched_quiery_inds, matched_label_inds = \
            assigner.assign(pred_instances, gt_instances)
        unique_quiery_inds = torch.unique(matched_quiery_inds)
        unique_label_inds = torch.unique(matched_label_inds)
        self.assertTrue(len(unique_quiery_inds) == len(matched_quiery_inds))
        self.assertTrue(
            torch.equal(unique_label_inds, torch.arange(0, num_classes)))

    def test_cls_match_cost(self):
        num_classes = 3
        num_masks = 10
        gt_instances = InstanceData()
        gt_instances.labels = torch.randint(0, num_classes, (num_classes, ))
        pred_instances = InstanceData()
        pred_instances.scores = torch.rand((num_masks, num_classes))

        # test ClassificationCost
        assigner = HungarianAssigner(dict(type='ClassificationCost'))
        matched_quiery_inds, matched_label_inds = \
            assigner.assign(pred_instances, gt_instances)
        unique_quiery_inds = torch.unique(matched_quiery_inds)
        unique_label_inds = torch.unique(matched_label_inds)
        self.assertTrue(len(unique_quiery_inds) == len(matched_quiery_inds))
        self.assertTrue(
            torch.equal(unique_label_inds, torch.arange(0, num_classes)))

    def test_mask_match_cost(self):
        num_classes = 3
        num_masks = 10
        num_points = 20
        gt_instances = InstanceData()
        gt_instances.masks = torch.randint(0, 2, (num_classes, num_points))
        pred_instances = InstanceData()
        pred_instances.masks = torch.rand((num_masks, num_points))

        # test DiceCost
        assigner = HungarianAssigner(
            dict(type='DiceCost', pred_act=True, eps=1.0))
        assign_result = assigner.assign(pred_instances, gt_instances)
        self.assertTrue(len(assign_result[0]) == len(assign_result[1]))

        # test CrossEntropyLossCost
        assigner = HungarianAssigner(
            dict(type='CrossEntropyLossCost', use_sigmoid=True))
        assign_result = assigner.assign(pred_instances, gt_instances)
        self.assertTrue(len(assign_result[0]) == len(assign_result[1]))
