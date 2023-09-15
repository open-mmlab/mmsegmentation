# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmengine.optim.scheduler import PolyLR

from mmseg.registry import PARAM_SCHEDULERS


@PARAM_SCHEDULERS.register_module()
class PolyLRRatio(PolyLR):
    """Implements polynomial learning rate decay with ratio.

    This scheduler adjusts the learning rate of each parameter group
    following a polynomial decay equation. The decay can occur in
    conjunction with external parameter adjustments made outside this
    scheduler.

    Args:
        optimizer (Optimizer or OptimWrapper): Wrapped optimizer.
        eta_min (float): Minimum learning rate at the end of scheduling.
            Defaults to 0.
        eta_min_ratio (float, optional): The ratio of the minimum parameter
            value to the base parameter value. Either `eta_min` or
            `eta_min_ratio` should be specified. Defaults to None.
        power (float): The power of the polynomial. Defaults to 1.0.
        begin (int): Step at which to start updating the parameters.
            Defaults to 0.
        end (int): Step at which to stop updating the parameters.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the value for each update.
            Defaults to False.
    """

    def __init__(self, eta_min_ratio: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eta_min_ratio = eta_min_ratio

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""

        if self.last_step == 0:
            return [
                group[self.param_name] for group in self.optimizer.param_groups
            ]

        param_groups_value = []
        for base_value, param_group in zip(self.base_values,
                                           self.optimizer.param_groups):
            eta_min = self.eta_min if self.eta_min_ratio is None else \
                base_value * self.eta_min_ratio
            step_ratio = (1 - 1 /
                          (self.total_iters - self.last_step + 1))**self.power
            step_value = (param_group[self.param_name] -
                          eta_min) * step_ratio + eta_min
            param_groups_value.append(step_value)

        return param_groups_value
