from config_build_data import ConfigBuildData as CFBD
from argument_handler import ArgumentHandler as ArgHand
from dict_utils import dataset_info
import dict_utils
from cfg_dict_generator import ConfigDictGenerator
from config_data_helper import ConfigDataHelper as CDH
import os
from mmengine.config import Config

from copy import deepcopy
from mmengine.runner import Runner

from mmseg.registry import RUNNERS
import torch

_empty_checkpoint = {
    "dataset_name"      :       None,
    "checkpoint_path"   :       None
}
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
    
def train_best(
    config_dir_path = "my_projects/configs",
):
    
    for cfg_name in os.listdir(config_dir_path):
        cfg_path = os.path.join(config_dir_path, cfg_name)
        cfg = Config.fromfile(cfg_path)
        cfg.work_dir = os.path.join('./work_dirs', cfg_name)
        
        
        try:
            run_cfg(cfg=cfg)
        except:
            print("failed")

def train_my_configs(
    config_dir_path = "my_projects/best_models/selection",
    datasets = ["irl_vision_sim", "hots-v1"],
    crop_size = (512, 512),
    iterations = 5000,
    save_interval = 1000,
    val_interval = 1000,
    unique = True,
    pretrained = True,
    save_best = True,
    exclude_names = []
):
    
    checkpoint_lookup_table = generate_checkpoint_lookup_by_cfg_name()
    
    # cfg_names = [
    #     "fpn_poolformer_m36_8xb4-40k_ade20k-512x512",
    #     "knet-s3_r50-d8_fcn_8xb2-adamw-80k_ade20k-512x512",
    #     "mask2former_r50_8xb2-160k_ade20k-512x512",
    #     "ocrnet_hr18s_4xb4-40k_voc12aug-512x512",
    #     "segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512",
    #     "twins_svt-b_fpn_fpnhead_8xb4-80k_ade20k-512x512"
    # ]
    expection_log_path = "my_projects/exception_log.log"
    for base_cfg_name in os.listdir(config_dir_path):
        # if len(b_data["cfg_name"]
        #     [name for name in exclude_names if name in base_cfg_name]
        # ) > 0:
        #     continue
    # for base_cfg_name in ["fcn_hr18s_4xb4-20k_voc12aug-512x512.py"]:
        base_cfg_path = os.path.join(config_dir_path, base_cfg_name)
        base_cfg_name = base_cfg_path.split("/")[-1].replace(".py", "")
        
        # if base_cfg_name not in cfg_names:
        #     continue
        # print(base_cfg_name)
        for target_dataset in datasets:
            checkpoints = [_empty_checkpoint]
            # TODO temp
            checkpoints = []
            if pretrained:
                checkpoints.append(
                    checkpoint_lookup_table[base_cfg_name]
                )  
            for checkpoint_dict in checkpoints:
                cfg_name = generate_new_cfg_name(
                    base_cfg_name=base_cfg_name,
                    dataset_name=target_dataset,
                    iterations=iterations,
                    crop_size=crop_size,
                    pretrain_data=checkpoint_dict["dataset_name"]
                )
                if unique and cfg_name in os.listdir("work_dirs"):
                    continue
                b_data = CFBD._get_cfg_build_data(
                    cfg_name=cfg_name, 
                    base_cfg_path=base_cfg_path,
                    dataset_cfg_path=dataset_info[target_dataset]["cfg_path"],
                    num_classes=dataset_info[target_dataset]["num_classes"],
                    pretrained= checkpoint_dict is not _empty_checkpoint, 
                    checkpoint_path=checkpoint_dict["checkpoint_path"],
                    pretrain_dataset=checkpoint_dict["dataset_name"], 
                    save_best=save_best, 
                    save_interval=save_interval,
                    val_interval=val_interval, 
                    batch_size=2, 
                    crop_size=crop_size,
                    iterations=iterations, 
                    epochs=None, 
                    dataset_name=target_dataset
                )
                cfg = ConfigDictGenerator._generate_config_from_build_data(cfg_build_data=b_data)
                # change_lr(cfg=cfg, lr=0.01)
                cfg.work_dir = os.path.join('./work_dirs', b_data["cfg_name"])
                try:
                    run_cfg(cfg=cfg)
                except Exception as exp:
                    
                    print(f"couldn't run config: {b_data['cfg_name']}")
                    print(f"excep:\n{exp}")
                    with open(expection_log_path, 'a') as ex_log:
                        log_line = f"cfg: {b_data['cfg_name']}\nexcept: {exp}"
                        ex_log.write(log_line)
                    
def change_lr(cfg, lr = 0.1, weight_decay = 0):
    cfg.optimizer["lr"] = lr
    cfg.optimizer["weight_decay"] = weight_decay
    cfg.optim_wrapper.optimizer["lr"] = lr
    cfg.optim_wrapper.optimizer["weight_decay"] = weight_decay
    
    
def generate_checkpoint_lookup_by_cfg_name(global_config_path = "configs"):
    
    lookup_table = {}
    for project_dir in os.listdir(global_config_path):
        meta_dict = CDH._read_metafile(
            project_name=project_dir,
            config_root_path=global_config_path
        )
        model_list = CDH._extract_model_list(metafile_dict=meta_dict)
        for model_dict in model_list:
            lookup_table[model_dict["name"]] = {
                "dataset_name"      :       model_dict["train_data"],
                "checkpoint_path"   :       model_dict["checkpoint_path"]
            }
    return lookup_table

def generate_new_cfg_name(
    base_cfg_name : str, 
    dataset_name : str, 
    iterations : int,
    crop_size : tuple,
    pretrain_data : str = "",
    device_info : str = "1xb2"
):
    # make new name 
    # basename with split at xb
    method_name = base_cfg_name.split("xb")[0][:-2]
    
    new_name = method_name + f"_{device_info}"
    if pretrain_data is not None and pretrain_data != "": 
        new_name += f'-pre-{pretrain_data.lower().replace(" ", "")}'
    new_name += f"-{int(iterations/1000)}k"
    new_name += f"_{dataset_name}"
    new_name += f"-{crop_size[0]}x{crop_size[1]}"
    return new_name


def test(
    config_dir_path = "configs/my_configs",
    datasets = ["irl_vision_sim", "hots-v1"],
    crop_size = (512, 512),
    iterations = 2000,
    save_interval = 500,
    val_interval = 500,
    unique = True,
    pretrained = True
):
    
    checkpoint_lookup_table = generate_checkpoint_lookup_by_cfg_name()
    
    for base_cfg_name in os.listdir(config_dir_path):
    # for base_cfg_name in ["fcn_hr18s_4xb4-20k_voc12aug-512x512.py"]:
        base_cfg_path = os.path.join(config_dir_path, base_cfg_name)
        base_cfg_name = base_cfg_path.split("/")[-1].replace(".py", "")
        # print(base_cfg_name)
        for target_dataset in datasets:
            checkpoints = [_empty_checkpoint]
            if pretrained:
                checkpoints.append(
                    checkpoint_lookup_table[base_cfg_name]
                )  
            for checkpoint_dict in checkpoints:
                cfg_name = generate_new_cfg_name(
                    base_cfg_name=base_cfg_name,
                    dataset_name=target_dataset,
                    iterations=iterations,
                    crop_size=crop_size,
                    pretrain_data=checkpoint_dict["dataset_name"]
                )
                if unique and cfg_name in os.listdir("work_dirs"):
                    continue
                b_data = CFBD._get_cfg_build_data(
                    cfg_name=cfg_name, 
                    base_cfg_path=base_cfg_path,
                    dataset_cfg_path=dataset_info[target_dataset]["cfg_path"],
                    num_classes=dataset_info[target_dataset]["num_classes"],
                    pretrained= checkpoint_dict is not _empty_checkpoint, 
                    checkpoint_path=checkpoint_dict["checkpoint_path"],
                    pretrain_dataset=checkpoint_dict["dataset_name"], 
                    save_best=False, 
                    save_interval=save_interval,
                    val_interval=val_interval, 
                    batch_size=2, 
                    crop_size=crop_size,
                    iterations=iterations, 
                    epochs=None, 
                    dataset_name=target_dataset
                )
                cfg = ConfigDictGenerator._generate_config_from_build_data(cfg_build_data=b_data)
                cfg.work_dir = os.path.join('./work_dirs', b_data["cfg_name"])
                try:
                    run_cfg(cfg=cfg)
                except:
                    print(f"couldn't run config: {b_data['cfg_name']}")


def main():
    # test()
    # train_my_configs()
    train_best()

if __name__ == '__main__':
    main()
