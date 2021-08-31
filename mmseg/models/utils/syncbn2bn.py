"""Modified from
https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547."""

from mmcv.utils.parrots_wrapper import SyncBatchNorm, _BatchNorm


class BatchNormXd(_BatchNorm):

    def _check_input_dim(self, input):
        # The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d,
        # etc is this method that is overwritten by the sub-class
        # This original goal of this method was for tensor sanity checks
        # If you're ok bypassing those sanity checks (eg. if you trust your
        # inference to provide the right dimensional inputs), then you can
        # just use this method for easy conversion from SyncBatchNorm
        # (unfortunately, SyncBatchNorm does not store the original class - if
        # it did we could return the one that was originally created)
        return


def revert_sync_batchnorm(module):
    # this is very similar to the function that it is trying to revert:
    # https://github.com/pytorch/pytorch/blob/c8b3686a3e4ba63dc59e5dcfe5db3430df256833/torch/nn/modules/batchnorm.py#L679
    module_output = module
    if isinstance(module, SyncBatchNorm):
        # to be consistent with SyncBN, we hack dim check function in BN
        module_output = BatchNormXd(module.num_features, module.eps,
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
