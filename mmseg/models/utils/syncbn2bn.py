"""Modified from
https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547."""

from mmcv.utils.parrots_wrapper import SyncBatchNorm, _BatchNorm


def revert_sync_batchnorm(module):
    # this is very similar to the function that it is trying to revert:
    # https://github.com/pytorch/pytorch/blob/c8b3686a3e4ba63dc59e5dcfe5db3430df256833/torch/nn/modules/batchnorm.py#L679
    module_output = module
    if isinstance(module, SyncBatchNorm):
        # to be consistent with SyncBN, we hack dim check function in BN
        module_output = _BatchNorm(module.num_features, module.eps,
                                   module.momentum, module.affine,
                                   module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output
