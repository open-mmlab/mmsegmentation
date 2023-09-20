# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from scipy.optimize import linear_sum_assignment
from torch.cuda.amp import autocast

from mmseg.registry import TASK_UTILS
from .base_assigner import BaseAssigner


@TASK_UTILS.register_module()
class HungarianAssigner(BaseAssigner):
    """Computes one-to-one matching between prediction masks and ground truth.

    This class uses bipartite matching-based assignment to computes an
    assignment between the prediction masks and the ground truth. The
    assignment result is based on the weighted sum of match costs. The
    Hungarian algorithm is used to calculate the best matching with the
    minimum cost. The prediction masks that are not matched are classified
    as background.

    Args:
        match_costs (ConfigDict|List[ConfigDict]): Match cost configs.
    """

    def __init__(
        self, match_costs: Union[List[Union[dict, ConfigDict]], dict,
                                 ConfigDict]
    ) -> None:

        if isinstance(match_costs, dict):
            match_costs = [match_costs]
        elif isinstance(match_costs, list):
            assert len(match_costs) > 0, \
                'match_costs must not be a empty list.'

        self.match_costs = [
            TASK_UTILS.build(match_cost) for match_cost in match_costs
        ]

    def assign(self, pred_instances: InstanceData, gt_instances: InstanceData,
               **kwargs):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The assignment first calculates the cost for each
        category assigned to each query mask, and then uses the
        Hungarian algorithm to calculate the minimum cost as the best
        match.

        Args:
            pred_instances (InstanceData): Instances of model
                predictions. It includes "masks", with shape
                (n, h, w) or (n, l), and "cls", with shape (n, num_classes+1)
            gt_instances (InstanceData): Ground truth of instance
                annotations. It includes "labels", with shape (k, ),
                and "masks", with shape (k, h, w) or (k, l).

        Returns:
            matched_quiery_inds (Tensor): The indexes of matched quieres.
            matched_label_inds (Tensor): The indexes of matched labels.
        """
        # compute weighted cost
        cost_list = []
        with autocast(enabled=False):
            for match_cost in self.match_costs:
                cost = match_cost(
                    pred_instances=pred_instances, gt_instances=gt_instances)
                cost_list.append(cost)
            cost = torch.stack(cost_list).sum(dim=0)

        device = cost.device
        # do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')

        matched_quiery_inds, matched_label_inds = linear_sum_assignment(cost)
        matched_quiery_inds = torch.from_numpy(matched_quiery_inds).to(device)
        matched_label_inds = torch.from_numpy(matched_label_inds).to(device)

        return matched_quiery_inds, matched_label_inds
