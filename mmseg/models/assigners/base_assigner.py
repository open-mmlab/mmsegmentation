# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Optional

from mmengine.structures import InstanceData


class BaseAssigner(metaclass=ABCMeta):
    """Base assigner that assigns masks to ground truth class labels."""

    @abstractmethod
    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs):
        """Assign masks to either a ground truth class label or a negative
        label."""
