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



config_bases =  {
                    "convnext"  :
                        {
                            "bases"             : 
                                [
                                    "convnext-tiny_upernet_8xb2-amp"
                                ],
                            "iterations"        :
                                [
                                    "1k", "5k", "10k", "20k", "40k", "80k", "160k"
                                ],
                            "datasets"          :
                                [
                                    "HOTS_v1-640x480"
                                ],
                            "pretrained_opts"   :
                                [
                                    "_pretrained_ade20k"
                                ]  
                        }
                        ,
                    "ddrnet"    :
                        {
                            "bases"             :
                                [
                                    "ddrnet_23-slim_in1k-pre_2xb6"
                                ],
                            "iterations"        :
                                [
                                    "1k", "5k", "10k", "20k", "40k", "80k", "160k"
                                ],
                            "datasets"          :
                                [
                                    "HOTS_v1-640x480"
                                ],
                            "pretrained_opts"   :
                                [
                                    "_pretrained_cityscapes"
                                ]
                        }
                        ,
                    "deeplabv3" :
                        {
                            "bases"             :
                                [
                                    "deeplabv3_r18-d8_4xb2",
                                    "deeplabv3_r18b-d8_4xb2"
                                ],
                            "iterations"        :
                                [
                                    "1k", "5k", "10k", "20k", "40k", "80k"
                                ],
                            "datasets"          :
                                [
                                    "HOTS_v1-640x480"
                                ],
                            "pretrained_opts"   :
                                [
                                    "_pretrained_cityscapes"
                                ]
                            
                            
                        }
                        ,
                    "deeplabv3plus" :
                        {
                            "bases"             :
                                [
                                    "deeplabv3plus_r18-d8_4xb2",
                                    "deeplabv3plus_r18b-d8_4xb2"
                                ],
                            "iterations"        :
                                [
                                    "1k", "5k", "10k", "20k", "40k", "80k"
                                ],
                            "datasets"          :
                                [
                                    "HOTS_v1-640x480"
                                ],
                            "pretrained_opts"   :
                                [
                                    "_pretrained_cityscapes"
                                ]
                            
                            
                        }
                        ,
                    "fastscnn"  : 
                        {
                            "bases"             :
                                [
                                    "fast_scnn_8xb4"
                                ],
                            "iterations"        :
                                [
                                    "1k", "5k", "10k", "20k", "40k", "80k", "160k"
                                ],
                            "datasets"          :
                                [
                                    "HOTS_v1-640x480"
                                ],
                            "pretrained_opts"   :
                                [
                                    "_pretrained_cityscapes"
                                ]
                            
                            
                        }
                        ,
                    "fcn"  : 
                        {
                            "bases"             :
                                [
                                    "fcn_r18-d8_4xb2",
                                    "fcn_r18b-d8_4xb2",
                                    "fcn-d6_r50-d16_4xb2",
                                    "fcn-d6_r50b-d16_4xb2",
                                    "fcn-d6_r101-d16_4xb2",
                                    "fcn-d6_r101b-d16_4xb2"                                   
                                ],
                            "iterations"        :
                                [
                                    "20k", "40k", "80k", "160k"
                                ],
                            "datasets"          :
                                [
                                    "HOTS_v1-640x480"
                                ],
                            "pretrained_opts"   :
                                [
                                    "_pretrained_cityscapes"
                                ]
                            
                            
                        }
                        ,
                    "hrnet"  : 
                        {
                            "bases"             :
                                [
                                    "fcn_hr18_4xb2",
                                    "fcn_hr18s_4xb2"
                                ],
                            "iterations"        :
                                [
                                    "20k", "40k", "80k", "160k"
                                ],
                            "datasets"          :
                                [
                                    "HOTS_v1-640x480"
                                ],
                            "pretrained_opts"   :
                                [
                                    "_pretrained_cityscapes"
                                ]
                            
                            
                        }
                        ,
                    "icnet"  : 
                        {
                            "bases"             :
                                [
                                    "icnet_r18-d8_4xb2",
                                    "icnet_r18-d8-in1k-pre_4xb2",
                                    "icnet_r50-d8_4xb2",
                                    "icnet_r50-d8-in1k-pre_4xb2",
                                    "icnet_r101-d8_4xb2",
                                    "icnet_r101-d8-in1k-pre_4xb2",                                  
                                ],
                            "iterations"        :
                                [
                                    "20k"
                                ],
                            "datasets"          :
                                [
                                    "HOTS_v1-640x480"
                                ],
                            "pretrained_opts"   :
                                [
                                    "_pretrained_cityscapes"
                                ]
                            
                            
                        }
                        ,
                    "mask2former"  : 
                        {
                            "bases"             :
                                [
                                    "mask2former_r50_8xb2",
                                    "mask2former_r101_8xb2",
                                    "mask2former_swin-t_8xb2"     
                                ],
                            "iterations"        :
                                [
                                    "20k", "40k", "80k", "160k"
                                ],
                            "datasets"          :
                                [
                                    # "HOTS_v1-640x480"
                                    "HOTS_v1-512x512"
                                ],
                            "pretrained_opts"   :
                                [
                                    "_pretrained_ade20k"
                                ]
                            
                            
                        }
                        ,
                    "maskformer"  : 
                        {
                            "bases"             :
                                [
                                    "maskformer_r50-d32_8xb2",
                                    "maskformer_r101-d32_8xb2",
                                    "maskformer_swin-t_upernet_8xb2",
                                    "maskformer_swin-s_upernet_8xb2"
                                    
                                ],
                            "iterations"        :
                                [
                                    "20k"
                                ],
                            "datasets"          :
                                [
                                    "HOTS_v1-512x512"
                                ],
                            "pretrained_opts"   :
                                [
                                    ""
                                ]
                            
                            
                        },
                    "segformer"  : 
                        {
                            "bases"             :
                                [
                                    "segformer_mit-b0_8xb2",
                                    "segformer_mit-b1_8xb2",
                                    "segformer_mit-b2_8xb2",
                                    "segformer_mit-b3_8xb2"
                                    
                                ],
                            "iterations"        :
                                [
                                    "20k"
                                ],
                            "datasets"          :
                                [
                                    "HOTS_v1-640x480"
    
                                ],
                            "pretrained_opts"   :
                                [
                                    ""
                                ]
                            
                            
                        }
                        ,
                    "segmenter"  : 
                        {
                            "bases"             :
                                [
                                    "segmenter_vit-t_mask_8xb1",
                                    "segmenter_vit-s_fcn_8xb1",
                                    "segmenter_vit-s_mask_8xb1",
                                    "segmenter_vit-b_mask_8xb1"
                                    
                                ],
                            "iterations"        :
                                [
                                    "20k"
                                ],
                            "datasets"          :
                                [
                                    "HOTS_v1-640x480"
    
                                ],
                            "pretrained_opts"   :
                                [
                                    ""
                                ]
                            
                            
                        }
                          
                }


def get_cfgs_model_list(args):
    cfgs_dict = dict()
    for model_name in args.models:
        if model_name not in list(config_bases.keys()):
            if args.verbose:
                print(f"model name not available in cfg_bases: {model_name}")
            continue
        cfgs_dict[model_name] = config_bases[model_name]
    for model, settings in cfgs_dict.items():
        iterations = list(set(settings["iterations"]).intersection(args.iterations))
        settings["iterations"] = iterations
        if args.pretrained:
            settings["pretrained_opts"].append("")
        else:
            settings["pretrained_opts"] = [""]
                   
    return cfgs_dict
        
def get_cfg_name_path_dict(cfgs_dict, verbose = False):
    cfg_name_path_dict = {"name" : [], "path" : []}
    for model, settings in cfgs_dict.items():    
        for base in settings["bases"]:
            for iteration in settings["iterations"]:
                for dataset in settings["datasets"]:
                    for pretrain_opt in settings["pretrained_opts"]:
                        cfg_name = f"{base}-{iteration}_{dataset}{pretrain_opt}.py"
                        cfg_path = os.path.join("configs", model, cfg_name)
                        if not os.path.exists(cfg_path):
                            if verbose:
                                print(f"path does not exist: {cfg_path}")
                            continue
                        cfg_name_path_dict["name"].append(cfg_name)
                        cfg_name_path_dict["path"].append(cfg_path)
    return cfg_name_path_dict   


def construct_custom_cfg_work_dir(cfg, cfg_path, args, dataset_name = "HOTS_v1"):
    old_cfg_name = osp.splitext(osp.basename(cfg_path))[0]
    cfg_name = construct_custom_cfg_name(cfg_name=old_cfg_name, args=args, dataset_name=dataset_name)
    cfg.work_dir = osp.join('./work_dirs',
                                cfg_name)
    # can't be changed bc properties
    # cfg.experiment_name = cfg_name
    # cfg.filename = cfg.filename.replace(old_cfg_name, cfg_name)
    
    print(cfg.filename, old_cfg_name, cfg_name)
    return cfg

# TODO kinda hardcoded
def construct_custom_cfg_name(cfg_name, args, dataset_name = "HOTS_v1"):
    name = copy.copy(cfg_name)
    if args.crop_size:
        crop = args.crop_size.lower()
        name = name.replace(name[name.find(dataset_name) + len(dataset_name) + 1 : name.find(dataset_name) + len(dataset_name) + 8], str(crop))
    return name
         
# only model that haven't been trained yet
def get_unique_cfg_name_path_dict(cfg_name_path_dict, args, work_dirs_path = "./work_dirs"):
    unique_cfg_name_path_dict = {"name" : [], "path" : []}
    for idx in range(len(cfg_name_path_dict["name"])):
        cfg_name = cfg_name_path_dict["name"][idx]
        if args.crop_size:
            cfg_name = construct_custom_cfg_name(cfg_name=cfg_name_path_dict["name"][idx], args=args)
        results_path = os.path.join(work_dirs_path, cfg_name.replace(".py", ""))
        if os.path.exists(results_path):
            continue
        unique_cfg_name_path_dict["name"].append(cfg_name_path_dict["name"][idx])
        unique_cfg_name_path_dict["path"].append(cfg_name_path_dict["path"][idx])
    return unique_cfg_name_path_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument(
                            "--models",
                            nargs='+',
                            default=list(config_bases.keys())
                        )
    parser.add_argument(
                            "--iterations",
                            nargs='+',
                            default=["20k", "40k", "80k", "160k"]
                        )
    parser.add_argument(
                            "--verbose",
                            action='store_true',
                            default=False
                        )
    
    parser.add_argument(
                            "--pretrained",
                            action='store_true',
                            default=False
                        )
    # parser.add_argument(
    #                         '--cfg-options',
    #                         nargs='+',
    #                         action=DictAction,
    #                         help='override some settings in the used config, the key-value pair '
    #                         'in xxx=yyy format will be merged into config file. If the value to '
    #                         'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
    #                         'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
    #                         'Note that the quotation marks are necessary and that no white space '
    #                         'is allowed.'
    # )
    parser.add_argument(
                            "--save_best",
                            action='store_true',
                            default=False
                        )
    parser.add_argument(
                            "--unique",
                            action='store_true',
                            default=False  
                        )
    parser.add_argument(
                            "--config",
                            nargs='+',
                            default=None  
                        )
    
    parser.add_argument(
                            "--save_interval",
                            type=int,
                            default=None
                        )
    parser.add_argument(
                            "--val_interval",
                            type=int,
                            default=None
                        )
    parser.add_argument(
                            "--batch_size",
                            type=int,
                            default=None,
                            help="train batch size"
                        )
    parser.add_argument(
                            "--crop_size",
                            type=str,
                            default=None,
                            help="crop size in #x# format"
                        )
    args = parser.parse_args() 
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

def main():
    args = parse_args()
    if args.verbose:
        print(f"args: {args}")
    # if args.config:
    #     for cfg_path in args.config:    
    #         cfg = Config.fromfile(cfg_path)
    #         run_cfg(cfg=cfg, cfg_path=cfg_path)
    #     exit()
    if args.config:
        cfg_name_path_dict = {"name" : [], "path" : []}
        for cfg_path in args.config:
            if not osp.exists(cfg_path):
                print(f"config path does not exist: {cfg_path}")
                continue
            cfg_name_path_dict["name"].append(cfg_path.split("/")[-1])
            cfg_name_path_dict["path"].append(cfg_path)
        print(cfg_name_path_dict)
        
    else:
        config_bases_selection = get_cfgs_model_list(args)
        
            
        cfg_name_path_dict = get_cfg_name_path_dict(cfgs_dict=config_bases_selection, verbose=args.verbose)
        
    if args.unique:
        cfg_name_path_dict = get_unique_cfg_name_path_dict(
                                                            cfg_name_path_dict=cfg_name_path_dict,
                                                            args=args,
                                                            work_dirs_path="./work_dirs"
                                                        )
    if args.verbose:
        print("configs: ")
        print('-' * 50)
        for idx in range(len(cfg_name_path_dict["name"])):
            print(f'task {idx}:\n  name: {cfg_name_path_dict["name"][idx]}\n  path: {cfg_name_path_dict["path"][idx]}')
            print('-' * 50)
    
    for idx in range(len(cfg_name_path_dict["name"])):
        if args.verbose:
            print(f'task {idx}: running cfg {cfg_name_path_dict["name"][idx]}')
        cfg = Config.fromfile(cfg_name_path_dict["path"][idx])
        
        additional_configs = []
        if args.save_best:
            additional_configs.append(
                    dict(
                        default_hooks = dict(
                                            checkpoint=dict(                                    
                                                            save_best='mIoU'
                                                            )
                                            )
                        )
            )
        
        if args.save_interval:
            additional_configs.append(
                dict(
                    default_hooks = dict(
                                        checkpoint=dict(                                    
                                                        interval=args.save_interval
                                                        )
                                        )
                            )
            )
        if args.val_interval:
            additional_configs.append(
                dict(
                    train_cfg = dict(
                                    val_interval=args.val_interval
                                    )
                )
            )
            
        if args.batch_size:
            additional_configs.append(
                dict(
                    train_dataloader = dict(
                        batch_size=args.batch_size
                    )
                )
            )
        if args.crop_size: 
            crop_size = tuple([int(size) for size in args.crop_size.lower().split('x')])
            data_preprocessor = dict(size=crop_size)
            # Load existing trainpipeline and change
            train_pipeline = cfg["train_pipeline"]
            for step_dict in train_pipeline:
                if step_dict["type"] == "RandomCrop":
                    step_dict["crop_size"] = crop_size
            
            additional_configs.append(
                dict(
                    crop_size=crop_size
                )
                    
                
            )
            additional_configs.append(
                dict(
                    data_preprocessor=data_preprocessor
                )
                    
                
            )
            additional_configs.append(
                dict(
                    model=dict(
                        data_preprocessor=data_preprocessor
                    )
                    
                )
            )
            additional_configs.append(
                dict(
                    train_pipeline=train_pipeline
                )
            )
            additional_configs.append(
                dict(
                    train_dataloader = dict(
                        dataset = dict(
                            pipeline=train_pipeline
                        )
                    )
                )
            )
                
        for additional_cfg in additional_configs:
            cfg.merge_from_dict(additional_cfg)
            
        cfg = construct_custom_cfg_work_dir(cfg=cfg, cfg_path=cfg_name_path_dict["path"][idx], args=args)
        run_cfg(cfg=cfg)
        



if __name__ == '__main__':
    main()