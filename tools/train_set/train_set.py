# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS
import torch
import copy
from dict_utils import (
    DEFAULT_CONFIG_ROOT_PATH, BATCH_SIZE_DEFAULT, N_GPU_DEFAULT,
    VAL_INTERVAL_EPOCH_DEFAULT, VAL_INTERVAL_ITERATIONS_DEFAULT,
    N_ITERATIONS_DEFAULT, dataset_info
)
from cfg_dict_generator import ConfigDictGenerator
from config_data_helper import ConfigDataHelper as CDHelper
from argument_handler import ArgumentHandler


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE_DEFAULT,
        help="train batch size"
    )
    
    parser.add_argument(
        "--crop_size",
        type=str,
        default="512x512",
        help="crop size in #x# format"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="hots-v1",
        choices=list(dataset_info.keys())
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None
    )
    parser.add_argument(
        "--trim_method_list",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--iterations",
        type=str,
        default=None,
        help="iterations in #k format"
    )
    parser.add_argument(
        "--projects",
        nargs='+',
        default=list(CDHelper._generate_available_project_list())
    )
    parser.add_argument(
        "--n_gpus",
        type=int,
        default=N_GPU_DEFAULT
    )
    parser.add_argument(
        "--pretrained",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--save_best",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=None
    )
    parser.add_argument(
        "--scratch",
        help="from scratch",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--unique",
        action='store_true',
        default=False  
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=None
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        default=False
    )
    
    args = parser.parse_args() 
    
    if not args.scratch and not args.pretrained:
        args.scratch = True
    
    args.crop_size = (int(args.crop_size.lower().split('x')[0]), int(args.crop_size.lower().split('x')[1]))
    if args.iterations:
        args.iterations = int(args.iterations.replace('k', '000')) if type(args.iterations) is str and 'k' in args.iterations else int(args.iterations)
        if not args.epochs and not args.val_interval:
            args.val_interval = VAL_INTERVAL_ITERATIONS_DEFAULT
            
    if args.epochs:
        args.iterations = None
        if not args.val_interval or args.val_interval > args.epochs:
            args.val_interval = VAL_INTERVAL_EPOCH_DEFAULT

    if not args.epochs and not args.iterations:
        args.iterations = N_ITERATIONS_DEFAULT
    return args

def run_cfg(cfg):
    torch.cuda.empty_cache()
    # load config
    
    cfg.launcher = "none"
    # work_dir is determined in this priority: CLI > segment in file > filename
    
    # maybe implement mixed precision training

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
    # start training
    runner.train()

# TODO  
def unique(cfg_build_data_list):
    cfg_build_data_list_ = []
    for cfg_bd in cfg_build_data_list:
        for cfg_bd_ in cfg_build_data_list_:
            if cfg_bd["cfg_name"] == cfg_bd_["cfg_name"]:
                continue
                   

def main():
    args = parse_args()
    if args.verbose:
        print(f"args: {args}")
        # method_list = ArgumentHandler._get_method_list_from_args(args=args)
        # for method in method_list:
        #     print(method)
    

        
    cfg_build_data_list = ArgumentHandler._generate_config_build_data_list(
        args=args
    )
    
    print(len(cfg_build_data_list))
    
   
    if args.verbose:
        for cfg_build_data in cfg_build_data_list:
            print(cfg_build_data["cfg_name"])
    #         print(cfg_build_data)
    # exit()
    # method_list = ArgumentHandler._get_method_list_from_args(args=args)  
    # for method in method_list:
    #     print('#' * 80)
    #     for key, val in method.items():
    #         print(f"{key} : {val}")  
    # exit()   
    for cfg_build_data in cfg_build_data_list:
        if args.verbose:
            print('#' * 80)
            print(f'running config: {cfg_build_data["cfg_name"]}')
            print('#' * 80)
        cfg = ConfigDictGenerator._generate_config_from_build_data(cfg_build_data=cfg_build_data)
        
        cfg.work_dir = osp.join('./work_dirs', cfg_build_data["cfg_name"])
        try:
            run_cfg(cfg=cfg)
        except:
            print(f"couldn't run config: {cfg_build_data['cfg_name']}")
    
    
if __name__ == '__main__':
    main()