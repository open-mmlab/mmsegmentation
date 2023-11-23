from pathlib import Path
from typing import List, Optional, Sequence, Union
from mmengine.hooks.hook import Hook
from mmengine.runner.runner import Runner
from mmengine.hooks.checkpoint_hook import CheckpointHook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class TestBestModelCheckpointHook(CheckpointHook):
    """Loads best known checkpoint before test loop."""

    def __init__(self,
        interval: int = -1,
        by_epoch: bool = True,
        save_optimizer: bool = True,
        save_param_scheduler: bool = True,
        out_dir: Union[str , Path , None] = None,
        max_keep_ckpts: int = -1,
        save_last: bool = True,
        save_best: Union[str , List[str] , None] = None,
        rule: Union[str , List[str] , None ] = None,
        greater_keys: Union[Sequence[str] , None ] = None,
        less_keys: Union[ Sequence[str] , None] = None,
        file_client_args: Union[dict , None ]= None,
        filename_tmpl:Union[ str , None ]= None,
        backend_args: Union[dict , None] = None,
        published_keys: Union[str , List[str] , None] = None,
        save_begin: int = 0,
        load_best_for_testing: Optional[str] = None,
        **kwargs) -> None:
        super().__init__(interval, by_epoch, save_optimizer, save_param_scheduler, out_dir, max_keep_ckpts, save_last, save_best, rule, greater_keys, less_keys, file_client_args, filename_tmpl, backend_args, published_keys, save_begin, **kwargs)

        if load_best_for_testing:
            assert load_best_for_testing in self.key_indicators
        self.load_best_for_testing = load_best_for_testing

    def before_test(self, runner: Runner) -> None:
        if not self.load_best_for_testing:
            return

        if len(self.key_indicators) == 1:
            best_checkpoint_path = self.best_ckpt_path
        else:
            best_checkpoint_path = self.best_ckpt_path_dict[self.load_best_for_testing]
        runner.load_checkpoint(filename=best_checkpoint_path)