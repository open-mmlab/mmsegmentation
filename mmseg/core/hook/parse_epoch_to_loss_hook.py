from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class ParseEpochToLossHook(Hook):
    """Check invalid loss hook.
    This hook will regularly check whether the loss is valid
    during training.
    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def before_train_epoch(self, runner):
        if hasattr(runner.model.module.decode_head.loss_decode, "epoch_num"):
            runner.model.module.decode_head.loss_decode.epoch_num = runner.epoch
            runner.model.module.decode_head.loss_decode.total_epochs = runner._max_epochs
