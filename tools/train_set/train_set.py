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
import dict_utils
from cfg_dict_generator import ConfigDictGenerator

BATCH_SIZE_DEFAULT = 2
N_GPU_DEFAULT = 1 
VAL_INTERVAL_EPOCH_DEFAULT = 1
VAL_INTERVAL_ITERATIONS_DEFAULT = 1000    

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE_DEFAULT,
        help="train batch size"
    )
    parser.add_argument(
        "--checkpoint",
        nargs='+',
        default=[]  
    )
    parser.add_argument(
        "--config",
        nargs='+',
        default=[]  
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
        default="HOTS_v1"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None
    )
    parser.add_argument(
        "--iterations",
        type=str,
        default=None,
        help="iterations in #k format"
    )
    parser.add_argument(
        "--models",
        nargs='+',
        default=list(dict_utils.config_bases.keys())
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
    
# TODO pretrained=True should run both pre trained and un trained
def from_cfg_list_arg(args):
    save_best = args.save_best
    save_interval = args.save_interval
    val_interval = args.val_interval
    batch_size = args.batch_size
    
    if len(args.checkpoint) == len(args.config):
        for idx in range(len(args.config)):
            cfg_name = args.config[idx].split('/')[-1].split('.')[0]
            base_cfg_path = args.config[idx]
            pretrained = args.pretrained
            checkpoint = args.checkpoint[idx]
            for iter in args.iterations:
                for crop_size in args.crop_sizes:
                    cfg_build_data = ConfigDictGenerator._get_cfg_build_data(
                        cfg_name=cfg_name, base_cfg_path=base_cfg_path, pretrained=pretrained,
                        checkpoint=checkpoint, save_best=save_best, save_interval=save_interval,
                        val_interval=val_interval, batch_size=batch_size, crop_size=crop_size,
                        iterations=iter
                    )
                    cfg = ConfigDictGenerator._generate_config_from_build_data(cfg_build_data=cfg_build_data)
                    cfg.work_dir = osp.join('./work_dirs', cfg_build_data["cfg_name"])
                    run_cfg(cfg=cfg)
            
        

def main():
    args = parse_args()
    if args.verbose:
        print(f"args: {args}")
        
    if args.config:
        from_cfg_list_arg(args=args)
            
    cfg_gen = ConfigDictGenerator()
    if args.verbose:
        cfg_name_list = cfg_gen.generate_config_names_list(args=args)
        print("cfg name list")
        for cfg_name in cfg_name_list:
            print(cfg_name)
        
    cfg_build_data_list = cfg_gen.generate_config_build_data_list(args=args)
    del cfg_gen
    for cfg_build_data in cfg_build_data_list:
        if args.verbose:
            print(f'running config: {cfg_build_data["cfg_name"]}')
        cfg = ConfigDictGenerator._generate_config_from_build_data(cfg_build_data=cfg_build_data)
        cfg.work_dir = osp.join('./work_dirs', cfg_build_data["cfg_name"])
        run_cfg(cfg=cfg)
    
    
if __name__ == '__main__':
    main()