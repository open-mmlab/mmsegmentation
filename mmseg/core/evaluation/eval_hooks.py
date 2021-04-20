import os.path as osp

from mmcv.runner import DistEvalHook as BasicDistEvalHook
from mmcv.runner import EvalHook as BasicEvalHook


class EvalHook(BasicEvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, efficient_test=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.efficient_test = efficient_test

    def after_train_iter(self, runner):
        """After train epoch hook."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        """After train epoch hook."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)


class DistEvalHook(BasicDistEvalHook):
    """Distributed EvalHook, with efficient test support.

    Args:
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, efficient_test=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.efficient_test = efficient_test

    def after_train_iter(self, runner):
        """After train epoch hook."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        """After train epoch hook."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)
