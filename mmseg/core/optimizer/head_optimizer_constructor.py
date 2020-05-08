import torch
from torch.nn import GroupNorm, LayerNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

from .builder import OPTIMIZER_BUILDERS
from .default_constructor import DefaultOptimizerConstructor


@OPTIMIZER_BUILDERS.register_module()
class HeadOptimizerConstructor(DefaultOptimizerConstructor):

    def add_params(self, params, module, prefix=''):
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
        """
        # get param-wise options
        bias_lr_mult = self.paramwise_cfg.get('bias_lr_mult', 1.)
        bias_decay_mult = self.paramwise_cfg.get('bias_decay_mult', 1.)
        norm_decay_mult = self.paramwise_cfg.get('norm_decay_mult', 1.)
        dwconv_decay_mult = self.paramwise_cfg.get('dwconv_decay_mult', 1.)
        decode_head_lr_mult = self.paramwise_cfg.get('decode_head_lr_mult', 1.)

        # special rules for norm layers and depth-wise conv layers
        is_norm = isinstance(module,
                             (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))
        is_dwconv = (
            isinstance(module, torch.nn.Conv2d)
            and module.in_channels == module.groups)

        for name, param in module.named_parameters(recurse=False):
            param_group = {'params': [param]}
            if not param.requires_grad:
                params.append(param_group)
                continue
            # bias_lr_mult affects all bias parameters except for norm.bias
            if name.endswith('bias') and not is_norm:
                param_group['lr'] = self.base_lr * bias_lr_mult
            # apply weight decay policies
            if self.base_wd is not None:
                # norm decay
                if is_norm:
                    param_group[
                        'weight_decay'] = self.base_wd * norm_decay_mult
                # depth-wise conv
                elif is_dwconv:
                    param_group[
                        'weight_decay'] = self.base_wd * dwconv_decay_mult
                # bias lr and decay
                elif name.endswith('bias'):
                    param_group[
                        'weight_decay'] = self.base_wd * bias_decay_mult
            if prefix.startswith('decode_head') and param.requires_grad:
                param_group['lr'] = self.base_lr * decode_head_lr_mult
            if prefix.startswith('auxiliary_head') and param.requires_grad:
                param_group['lr'] = self.base_lr * decode_head_lr_mult
            params.append(param_group)

        for child_name, child_mod in module.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            self.add_params(params, child_mod, prefix=child_prefix)
