# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmseg.registry import TASK_UTILS

PIXEL_SAMPLERS = TASK_UTILS


def build_pixel_sampler(cfg, **default_args):
    """Build pixel sampler for segmentation map."""
    warnings.warn(
        '``build_pixel_sampler`` would be deprecated soon, please use '
        '``mmseg.registry.TASK_UTILS.build()`` ')
    return TASK_UTILS.build(cfg, default_args=default_args)
