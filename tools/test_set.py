# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
import torch
from multiprocessing import Pool, cpu_count
from copy import deepcopy

def fix_test_loader(cfg, dataset = "evaltest"):
    img_pth = os.path.join("img_dir", dataset)
    seg_map_path = os.path.join("ann_dir", dataset)
    cfg.test_dataloader.dataset.data_prefix["img_path"] = img_pth
    cfg.test_dataloader.dataset.data_prefix["seg_map_path"] = seg_map_path
    return cfg

    
def trigger_visualization_hook(cfg_dict, show_dir):
    cfg_dict["default_hooks"] = dict(
        checkpoint=dict(by_epoch=False, interval=1000, type='CheckpointHook'),
        logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
        param_scheduler=dict(type='ParamSchedulerHook'),
        sampler_seed=dict(type='DistSamplerSeedHook'),
        timer=dict(type='IterTimerHook'),
        visualization=dict(type='SegVisualizationHook', draw = True)
    )
    
    cfg_dict["vis_backends"] = [
        dict(type='LocalVisBackend')
    ]
    cfg_dict["visualizer"] = dict(
        name='visualizer',
        type='SegLocalVisualizer',
        save_dir=show_dir,
        vis_backends=[
            dict(type='LocalVisBackend'),
        ]
    )
    return cfg_dict



# def trigger_visualization_hook(cfg, args):
#     default_hooks = cfg.default_hooks
#     if 'visualization' in default_hooks:
#         visualization_hook = default_hooks['visualization']
#         # Turn on visualization
#         visualization_hook['draw'] = True
#         if args.show:
#             visualization_hook['show'] = True
#             visualization_hook['wait_time'] = args.wait_time
#         if args.show_dir:
#             visualizer = cfg.visualizer
#             visualizer['save_dir'] = args.show_dir
#     else:
#         raise RuntimeError(
#             'VisualizationHook must be included in default_hooks.'
#             'refer to usage '
#             '"visualization=dict(type=\'VisualizationHook\')"')

#     return cfg


    
# TODO doesnt work
def fix_log_paths(
    test_results_path = "test_results"
    ):
    for project_name in os.listdir(test_results_path):
        project_path = os.path.join(test_results_path, project_name)
        for experiment_name in os.listdir(project_path): # iter
            experiment_path = os.path.join(project_path, experiment_name)
            for sub_experiment_name in os.listdir(experiment_path): # timestamp dir
                sub_experiment_path = os.path.join(experiment_path, sub_experiment_name)
                log_file = [file for file in os.listdir(sub_experiment_path) if ".log" in file][0]
                log_file_path = os.path.join(sub_experiment_path, log_file)
                if os.path.exists(log_file_path):
                    os.rename(log_file_path, os.path.join(experiment_path, log_file))

def trimmed_projects(
    exclude = [], 
    unique = True, work_dir_path = "work_dirs",
    test_results_path = "test_results",
    training_iters = None
    ):
    def get_iters(project_name):
        train_set = project_name.split("_")[-2]
        iters = train_set.split("-")[-1]
        if 'k' in iters:
            iters = iters.replace('k', '000')
        return int(iters)
    
    project_names = os.listdir(work_dir_path)
    tested_project_names = os.listdir(test_results_path)
    project_names = [project_name for project_name in project_names if project_names not in exclude]
    if training_iters is not None:
        project_names = [project_name for project_name in project_names if training_iters == get_iters(project_name)]
    if unique:
        project_names = [project_name for project_name in project_names if project_name not in tested_project_names]
    # # TODO temp:
    # project_names = ["maskformer_r50-d32_1xb2-pre-ade20k-1k_hots-v1-512x512"]
    
    return project_names

def trimmed_checkpoints(
    project_name, work_dir_path = "work_dirs",
    test_results_path = "test_results", 
    exlude = [], unique = True
    ):
    project_path = os.path.join(work_dir_path, project_name)
    checkpoint_names = [file_name for file_name in os.listdir(project_path) if ".pth" in file_name]
    test_work_dir_path = os.path.join(test_results_path, project_name)
    if not os.path.exists(test_work_dir_path):
        return checkpoint_names
    tested_checkpoints = os.listdir(test_work_dir_path)
    if unique:
        checkpoint_names = [
            checkpoint_name for checkpoint_name in checkpoint_names
                if checkpoint_name not in tested_checkpoints
        ]
    return checkpoint_names
    
def main():
    test_results_path = "test_results"
    work_dir_path = "work_dirs"
    one_test = False
    # unique false bc one_test True
    project_names = os.listdir(work_dir_path)
    # project_names = trimmed_projects(
    #     work_dir_path=work_dir_path, 
    #     test_results_path=test_results_path,
    #     unique=(not one_test)
    # )
    print(project_names)
    
    for project_name in project_names:
        print(f"evaluating project: {project_name}")
        project_path = os.path.join(work_dir_path, project_name)
        # checkpoint_names = trimmed_checkpoints(
        #     project_name=project_name,
        #     work_dir_path=work_dir_path,
        #     test_results_path=test_results_path
        # )
        
        checkpoint_names = [    
            file for file in os.listdir(project_path) if ".pth" in file
        ]
        for checkpoint_name in checkpoint_names:
            config_name = [file_name for file_name in os.listdir(project_path) if ".py" in file_name][0]
            config_path = os.path.join(project_path, config_name)
            test_work_dir_path = os.path.join(test_results_path, project_name)
            
            cfg_dict = Config.fromfile(config_path).to_dict()
            output_dir = os.path.join(test_work_dir_path, checkpoint_name, "out")
            show_dir = os.path.join(test_work_dir_path, checkpoint_name, "show")
            
            cfg_dict = trigger_visualization_hook(cfg_dict, show_dir)
            cfg = Config(cfg_dict=cfg_dict)
            cfg.test_evaluator["output_dir"] = output_dir
            cfg.test_evaluator["keep_results"] = True
            torch.cuda.empty_cache()
            
            cfg.work_dir = os.path.join(test_work_dir_path, checkpoint_name)
            checkpoint_path = os.path.join(project_path, checkpoint_name)
            cfg.load_from = checkpoint_path
            cfg.test_evaluator = dict(type="IoUMetric")
            # cfg.test_evaluator = dict(type="IoUMetricFixed")
            
            # TEMP when using test eval merged dataset
            # cfg = fix_test_loader(cfg=cfg, dataset="evaltest")
            
            # TODO temp####################################################
            
            ##################################################################
            try:
                runner = Runner.from_cfg(cfg=cfg)
                
                runner.test()
            except:
                print(f"cfg: {cfg.work_dir} did not work")
            # TODO temp  Somehow I cant make show work after first iteration
            torch.cuda.empty_cache()
            del cfg
            del cfg_dict
            if one_test:
                exit()
            
            
if __name__ == '__main__':
    main()
