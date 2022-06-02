# Copyright (c) OpenMMLab. All rights reserved.
import copy

from mmseg.registry import OPTIM_WRAPPER_CONSTRUCTORS


def build_optimizer(model, cfg):
    optim_wrapper_cfg = copy.deepcopy(cfg)
    constructor_type = optim_wrapper_cfg.pop('constructor',
                                             'DefaultOptimWrapperConstructor')
    paramwise_cfg = optim_wrapper_cfg.pop('paramwise_cfg', None)
    optim_wrapper_builder = OPTIM_WRAPPER_CONSTRUCTORS.build(
        dict(
            type=constructor_type,
            optim_wrapper_cfg=optim_wrapper_cfg,
            paramwise_cfg=paramwise_cfg))
    optim_wrapper = optim_wrapper_builder(model)
    return optim_wrapper
