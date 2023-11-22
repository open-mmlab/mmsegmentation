
from typing import Optional, Sequence
from mmengine.hooks.logger_hook import DATA_BATCH, LoggerHook
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmseg.registry import HOOKS



# Force test loop before logger hook cleans up visualizer...
@HOOKS.register_module()
class ForceRunTestLoop(Hook):

    priority = "HIGHEST"

    def after_run(self, runner: Runner) -> None:

        if runner._test_loop is None:
            raise RuntimeError(
                '`self._test_loop` should not be None when calling test '
                'method. Please provide `test_dataloader`, `test_cfg` and '
                '`test_evaluator` arguments when initializing runner.')
        runner._test_loop = runner.build_test_loop(runner._test_loop) 
        runner.test_loop.run()


