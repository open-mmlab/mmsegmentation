# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn
from mmengine.logging import print_log
from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.utils.dl_utils import mmcv_full_available
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm, _InstanceNorm
from torch.nn import GroupNorm, LayerNorm

from mmseg.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class ForceDefaultOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    """Default constructor with forced optimizer settings.

    This constructor extends the default constructor to add an option for
    forcing default optimizer settings. This is useful for ensuring that
    certain parameters or layers strictly adhere to pre-defined default
    settings, regardless of any custom settings specified.

    By default, each parameter share the same optimizer settings, and we
    provide an argument ``paramwise_cfg`` to specify parameter-wise settings.
    It is a dict and may contain various fields like 'custom_keys',
    'bias_lr_mult', etc., as well as the additional field
    `force_default_settings` which allows for enforcing default settings on
    optimizer parameters.

    - ``custom_keys`` (dict): Specified parameters-wise settings by keys. If
      one of the keys in ``custom_keys`` is a substring of the name of one
      parameter, then the setting of the parameter will be specified by
      ``custom_keys[key]`` and other setting like ``bias_lr_mult`` etc. will
      be ignored. It should be noted that the aforementioned ``key`` is the
      longest key that is a substring of the name of the parameter. If there
      are multiple matched keys with the same length, then the key with lower
      alphabet order will be chosen.
      ``custom_keys[key]`` should be a dict and may contain fields ``lr_mult``
      and ``decay_mult``. See Example 2 below.
    - ``bias_lr_mult`` (float): It will be multiplied to the learning
      rate for all bias parameters (except for those in normalization
      layers and offset layers of DCN).
    - ``bias_decay_mult`` (float): It will be multiplied to the weight
      decay for all bias parameters (except for those in
      normalization layers, depthwise conv layers, offset layers of DCN).
    - ``norm_decay_mult`` (float): It will be multiplied to the weight
      decay for all weight and bias parameters of normalization
      layers.
    - ``flat_decay_mult`` (float): It will be multiplied to the weight
      decay for all one-dimensional parameters
    - ``dwconv_decay_mult`` (float): It will be multiplied to the weight
      decay for all weight and bias parameters of depthwise conv
      layers.
    - ``dcn_offset_lr_mult`` (float): It will be multiplied to the learning
      rate for parameters of offset layer in the deformable convs
      of a model.
    - ``bypass_duplicate`` (bool): If true, the duplicate parameters
      would not be added into optimizer. Defaults to False.
    - ``force_default_settings`` (bool): If true, this will override any
      custom settings defined by ``custom_keys`` and enforce the use of
      default settings for optimizer parameters like ``bias_lr_mult``.
      This is particularly useful when you want to ensure that certain layers
      or parameters adhere strictly to the pre-defined default settings.

    Note:

        1. If the option ``dcn_offset_lr_mult`` is used, the constructor will
        override the effect of ``bias_lr_mult`` in the bias of offset layer.
        So be careful when using both ``bias_lr_mult`` and
        ``dcn_offset_lr_mult``. If you wish to apply both of them to the offset
        layer in deformable convs, set ``dcn_offset_lr_mult`` to the original
        ``dcn_offset_lr_mult`` * ``bias_lr_mult``.

        2. If the option ``dcn_offset_lr_mult`` is used, the constructor will
        apply it to all the DCN layers in the model. So be careful when the
        model contains multiple DCN layers in places other than backbone.

        3. When the option ``force_default_settings`` is true, it will override
        any custom settings provided in ``custom_keys``. This ensures that the
        default settings for the optimizer parameters are used.

    Args:
        optim_wrapper_cfg (dict): The config dict of the optimizer wrapper.

            Required fields of ``optim_wrapper_cfg`` are

            - ``type``: class name of the OptimizerWrapper
            - ``optimizer``: The configuration of optimizer.

            Optional fields of ``optim_wrapper_cfg`` are

            - any arguments of the corresponding optimizer wrapper type,
              e.g., accumulative_counts, clip_grad, etc.

            Required fields of ``optimizer`` are

            - `type`: class name of the optimizer.

            Optional fields of ``optimizer`` are

            - any arguments of the corresponding optimizer type, e.g.,
              lr, weight_decay, momentum, etc.

        paramwise_cfg (dict, optional): Parameter-wise options.

    Example 1:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> optim_wrapper_cfg = dict(
        >>>     dict(type='OptimWrapper', optimizer=dict(type='SGD', lr=0.01,
        >>>         momentum=0.9, weight_decay=0.0001))
        >>> paramwise_cfg = dict(norm_decay_mult=0.)
        >>> optim_wrapper_builder = DefaultOptimWrapperConstructor(
        >>>     optim_wrapper_cfg, paramwise_cfg)
        >>> optim_wrapper = optim_wrapper_builder(model)

    Example 2:
        >>> # assume model have attribute model.backbone and model.cls_head
        >>> optim_wrapper_cfg = dict(type='OptimWrapper', optimizer=dict(
        >>>     type='SGD', lr=0.01, weight_decay=0.95))
        >>> paramwise_cfg = dict(custom_keys={
        >>>     'backbone': dict(lr_mult=0.1, decay_mult=0.9)})
        >>> optim_wrapper_builder = DefaultOptimWrapperConstructor(
        >>>     optim_wrapper_cfg, paramwise_cfg)
        >>> optim_wrapper = optim_wrapper_builder(model)
        >>> # Then the `lr` and `weight_decay` for model.backbone is
        >>> # (0.01 * 0.1, 0.95 * 0.9). `lr` and `weight_decay` for
        >>> # model.cls_head is (0.01, 0.95).
    """

    def add_params(self,
                   params: List[dict],
                   module: nn.Module,
                   prefix: str = '',
                   is_dcn_module: Optional[Union[int, float]] = None) -> None:
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        # get param-wise options
        custom_keys = self.paramwise_cfg.get('custom_keys', {})
        # first sort with alphabet order and then sort with reversed len of str
        sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)

        bias_lr_mult = self.paramwise_cfg.get('bias_lr_mult', None)
        bias_decay_mult = self.paramwise_cfg.get('bias_decay_mult', None)
        norm_decay_mult = self.paramwise_cfg.get('norm_decay_mult', None)
        dwconv_decay_mult = self.paramwise_cfg.get('dwconv_decay_mult', None)
        flat_decay_mult = self.paramwise_cfg.get('flat_decay_mult', None)
        bypass_duplicate = self.paramwise_cfg.get('bypass_duplicate', False)
        dcn_offset_lr_mult = self.paramwise_cfg.get('dcn_offset_lr_mult', None)
        force_default_settings = self.paramwise_cfg.get(
            'force_default_settings', False)

        # special rules for norm layers and depth-wise conv layers
        is_norm = isinstance(module,
                             (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))
        is_dwconv = (
            isinstance(module, torch.nn.Conv2d)
            and module.in_channels == module.groups)

        for name, param in module.named_parameters(recurse=False):
            param_group = {'params': [param]}
            if bypass_duplicate and self._is_in(param_group, params):
                print_log(
                    f'{prefix} is duplicate. It is skipped since '
                    f'bypass_duplicate={bypass_duplicate}',
                    logger='current',
                    level=logging.WARNING)
                continue
            if not param.requires_grad:
                params.append(param_group)
                continue

            # if the parameter match one of the custom keys, ignore other rules
            is_custom = False
            for key in sorted_keys:
                if key in f'{prefix}.{name}':
                    is_custom = True
                    lr_mult = custom_keys[key].get('lr_mult', 1.)
                    param_group['lr'] = self.base_lr * lr_mult
                    if self.base_wd is not None:
                        decay_mult = custom_keys[key].get('decay_mult', 1.)
                        param_group['weight_decay'] = self.base_wd * decay_mult
                    # add custom settings to param_group
                    for k, v in custom_keys[key].items():
                        param_group[k] = v
                    break

            if not is_custom or force_default_settings:
                # bias_lr_mult affects all bias parameters
                # except for norm.bias dcn.conv_offset.bias
                if name == 'bias' and not (
                        is_norm or is_dcn_module) and bias_lr_mult is not None:
                    param_group['lr'] = self.base_lr * bias_lr_mult

                if (prefix.find('conv_offset') != -1 and is_dcn_module
                        and dcn_offset_lr_mult is not None
                        and isinstance(module, torch.nn.Conv2d)):
                    # deal with both dcn_offset's bias & weight
                    param_group['lr'] = self.base_lr * dcn_offset_lr_mult

                # apply weight decay policies
                if self.base_wd is not None:
                    # norm decay
                    if is_norm and norm_decay_mult is not None:
                        param_group[
                            'weight_decay'] = self.base_wd * norm_decay_mult
                    # bias lr and decay
                    elif (name == 'bias' and not is_dcn_module
                          and bias_decay_mult is not None):
                        param_group[
                            'weight_decay'] = self.base_wd * bias_decay_mult
                    # depth-wise conv
                    elif is_dwconv and dwconv_decay_mult is not None:
                        param_group[
                            'weight_decay'] = self.base_wd * dwconv_decay_mult
                    # flatten parameters except dcn offset
                    elif (param.ndim == 1 and not is_dcn_module
                          and flat_decay_mult is not None):
                        param_group[
                            'weight_decay'] = self.base_wd * flat_decay_mult
            params.append(param_group)
            for key, value in param_group.items():
                if key == 'params':
                    continue
                full_name = f'{prefix}.{name}' if prefix else name
                print_log(
                    f'paramwise_options -- {full_name}:{key}={value}',
                    logger='current')

        if mmcv_full_available():
            from mmcv.ops import DeformConv2d, ModulatedDeformConv2d
            is_dcn_module = isinstance(module,
                                       (DeformConv2d, ModulatedDeformConv2d))
        else:
            is_dcn_module = False
        for child_name, child_mod in module.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            self.add_params(
                params,
                child_mod,
                prefix=child_prefix,
                is_dcn_module=is_dcn_module)
