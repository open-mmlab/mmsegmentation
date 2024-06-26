from config_build_data import ConfigBuildData as CFBD
from argument_handler import ArgumentHandler as ArgHand
from dict_utils import dataset_info
import dict_utils
from cfg_dict_generator import ConfigDictGenerator
import os
from mmengine.config import Config

from copy import deepcopy
from mmengine.runner import Runner

from mmseg.registry import RUNNERS
import torch

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
    
def get_my_models(
    my_models_path = "my_projects/best_models/hots_v1", 
    target_dataset = "irl_vision_sim"
):
    config_build_data_list = []
    for model_dir_name in os.listdir(my_models_path):
        model_dir_path = os.path.join(my_models_path, model_dir_name)
        cfg_name = model_dir_name.replace("hots-v1", target_dataset)
        cfg_name = cfg_name.replace("-1k_", "-2k_")
        
        base_cfg = [
            file_ for file_ in os.listdir(model_dir_path) if '.py' in file_
        ][0]
        
        base_cfg_path = os.path.join(model_dir_path, base_cfg)
        b_data = CFBD._get_cfg_build_data(
            cfg_name=cfg_name, base_cfg_path=base_cfg_path,
            dataset_cfg_path=dataset_info[target_dataset]["cfg_path"],
            num_classes=dataset_info[target_dataset]["num_classes"],
            pretrained=False, checkpoint_path=None,
            pretrain_dataset=None, save_best=False, save_interval=500,
            val_interval=500, batch_size=None, crop_size=(512, 512),
            iterations=2000, epochs=None, dataset_name=target_dataset
        )
        cfg = ConfigDictGenerator._generate_config_from_build_data(cfg_build_data=b_data)
        cfg.work_dir = os.path.join('./my_projects/work_dirs', b_data["cfg_name"])
        try:
            run_cfg(cfg=cfg)
        except:
            print(f"couldn't run config: {b_data['cfg_name']}")
    

# def apply_dataset(dataset_info, cfg_name, new_cfg: Config):
#     num_classes = dataset_info["num_classes"]
#     class_weight = dataset_info["class_weight"]
#     if "mask" in cfg_name and "former" in cfg_name:
#         class_weight = [0.1] + ([1.0] * num_classes)
#     dataset_cfg = Config.fromfile(
#         dataset_info["cfg_path"]
#     )
#     for key, value in dataset_cfg.items():
#         new_cfg[key] = value
#     dict_utils.BFS_change_key(
#         cfg=new_cfg, 
#         target_key="num_classes", 
#         new_value=num_classes
#     )
#     dict_utils.BFS_change_key(
#         cfg=new_cfg,
#         target_key="class_weight",
#         new_value=class_weight
#     )

def main():
    get_my_models()


if __name__ == '__main__':
    main()
# def get_my_models(
#     args,
#     my_models_path = "my_projects/best_models/hots_v1", 
#     target_data_set = "irl_vision_sim"
# ):
#     config_build_data_list = []
#     for model_dir_name in os.listdir(my_models_path):
#         model_dir_path = os.path.join(my_models_path, model_dir_name)
#         model_dict = {
#             "checkpoint_path"       :       os.path.join(
#                                                 model_dir_path, "iter_500.pth")
#         }
#         checkpoint_paths = ArgHand._get_checkpoint_paths(
#             args=args, model_dict=model_dict
#         )
#         for checkpoint_pth in checkpoint_paths:
#             build_dict = CFBD._get_empty_cfg_build_data()
#             pre_train_data = ""
#             if "-pre-" in model_dir_name:
#                 name = model_dir_name
#                 pre_train_data = name[
#                     name.index("-pre-"):].replace("-pre-","").split("-")[0]

#             build_dict["cfg_name"] = ArgHand._generate_config_name(
#                 args=args, model_dict={}, 
#                 method_name=model_dir_name.split('1xb')[0],
#                 pretrained=bool(checkpoint_pth), pretrain_data=pre_train_data
                
#             )
#             base_cfg = [
#                 file_ for file_ in os.listdir(model_dir_path) if '.py' in file_
#             ][0]
            
#             build_dict["base_cfg_path"] = os.path.join(model_dir_path, base_cfg)
#             build_dict["dataset_cfg_path"] = dataset_info[args.dataset]
            
    
# base_cfg = Config.fromfile(filename=cfg_build_data["base_cfg_path"])
# new_cfg_dict = base_cfg.to_dict()
# new_cfg = Config(cfg_dict=new_cfg_dict) 

# additional_configs = []

# ConfigDictGenerator.apply_dataset(
#     cfg_build_data=cfg_build_data,
#     new_cfg=new_cfg
# )

# def apply_dataset(cfg_build_data: dict, new_cfg: Config):
#     dataset_info = dataset_info[cfg_build_data["dataset"]]
#     num_classes = dataset_info["num_classes"]
#     class_weight = dataset_info["class_weight"]
#     if "mask" in cfg_build_data["cfg_name"] and "former" in cfg_build_data["cfg_name"]:
#         class_weight = [0.1] + ([1.0] * num_classes)
#     dataset_cfg = Config.fromfile(
#         dataset_info["cfg_path"]
#     )
#     for key, value in dataset_cfg.items():
#         new_cfg[key] = value
#     dict_utils.BFS_change_key(
#         cfg=new_cfg, 
#         target_key="num_classes", 
#         new_value=num_classes
#     )
#     dict_utils.BFS_change_key(
#         cfg=new_cfg,
#         target_key="class_weight",
#         new_value=class_weight
#     )
