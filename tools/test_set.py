# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
import torch





def trigger_visualization_hook(cfg, show_dir):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        
        visualizer = cfg.visualizer
        visualizer['save_dir'] = show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg
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
    # project_names = ["convnext-tiny_upernet_1xb2-300_hots-v1-512x512"]
    return project_names

def main():
    test_results_path = "test_results"
    work_dir_path = "work_dirs"
    project_names = trimmed_projects(work_dir_path=work_dir_path)
    print(project_names)
    
    for project_name in project_names:
        print(f"evaluating project: {project_name}")
        project_path = os.path.join(work_dir_path, project_name)
        config_name = [file_name for file_name in os.listdir(project_path) if ".py" in file_name][0]
        checkpoint_names = [file_name for file_name in os.listdir(project_path) if ".pth" in file_name]
        config_path = os.path.join(project_path, config_name)
        
        cfg = Config.fromfile(config_path)
        test_work_dir_path = os.path.join(test_results_path, project_name)
        cfg.work_dir = test_work_dir_path
        for checkpoint_name in checkpoint_names:
            torch.cuda.empty_cache()
            cfg.work_dir = os.path.join(test_work_dir_path, checkpoint_name)
            checkpoint_path = os.path.join(project_path, checkpoint_name)
            cfg.load_from = checkpoint_path
            cfg.test_evaluator = dict(type="IoUMetricFixed")
            # TODO temp####################################################
            output_dir = os.path.join(test_work_dir_path, checkpoint_name, "out")
            show_dir = os.path.join(test_work_dir_path, checkpoint_name, "show")
            cfg.test_evaluator["output_dir"] = output_dir
            cfg.test_evaluator["keep_results"] = True
            trigger_visualization_hook(cfg, show_dir=show_dir)
            ##################################################################
            try:
                runner = Runner.from_cfg(cfg=cfg)
                
                runner.test()
            except:
                print(f"cfg: {cfg.work_dir} did not work")
            
            
if __name__ == '__main__':
    main()
