import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS
import torch

def run_cfg(cfg, cfg_path):
    torch.cuda.empty_cache()
    # load config
    
    cfg.launcher = "none"
    
    # work_dir is determined in this priority: CLI > segment in file > filename
    
    cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(cfg_path))[0] + "_test")

    # maybe implement mixed precision training

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        print("default runner")
        # runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        print("runner type set")
        # runner = RUNNERS.build(cfg)

    # start training
    # runner.train()
    
def main():
    cfg_path = "./configs/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-10k_HOTS_v1-640x480.py"
    cfg = Config.fromfile(cfg_path)
    print(cfg.filename)
    
if __name__ == '__main__':
    main()