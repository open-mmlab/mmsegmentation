import os.path as osp

import torch.distributed as dist
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm


class EvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)

    def progressive_evaluate(self, runner, processor):
        """Evaluate the results by progressive mode.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            processor (object): Output processor.
        """
        eval_res = self.dataloader.dataset.progressive_evaluate(
            processor, logger=runner.logger, **self.eval_kwargs)

        # TODO: Blocked by mmcv pr: #1213
        # evaluation info specific buffer
        # runner.log_buffer.output['eval_res'] = {}
        # for name, val in eval_res.items():
        #     runner.log_buffer.output['eval_res'][name] = val
        runner.log_buffer.clear()
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

        if self.save_best is not None:
            if self.key_indicator == 'auto':
                # infer from eval_results
                self._init_rule(self.rule, list(eval_res.keys())[0])
            return eval_res[self.key_indicator]

        return None

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmseg.apis import progressive_single_gpu_test
        processor = progressive_single_gpu_test(
            runner.model, self.dataloader, False, show=False)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.progressive_evaluate(runner, processor)
        if self.save_best:
            self._save_ckpt(runner, key_score)


class DistEvalHook(_DistEvalHook):
    """Distributed EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)

    def progressive_evaluate(self, runner, processor):
        """Evaluate the results by progressive mode.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            processor (object): Output processor.
        """
        eval_res = self.dataloader.dataset.progressive_evaluate(
            processor, logger=runner.logger, **self.eval_kwargs)
        # TODO: Blocked by mmcv pr: #1213
        # evaluation info specific buffer
        # runner.log_buffer.output['eval_res'] = {}
        # for name, val in eval_res.items():
        #     runner.log_buffer.output['eval_res'][name] = val
        runner.log_buffer.clear()
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

        if self.save_best is not None:
            if self.key_indicator == 'auto':
                # infer from eval_results
                self._init_rule(self.rule, list(eval_res.keys())[0])
            return eval_res[self.key_indicator]

        return None

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmseg.apis import progressive_multi_gpu_test
        processor = progressive_multi_gpu_test(
            runner.model,
            self.dataloader,
            False,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.progressive_evaluate(runner, processor)

            if self.save_best:
                self._save_ckpt(runner, key_score)
