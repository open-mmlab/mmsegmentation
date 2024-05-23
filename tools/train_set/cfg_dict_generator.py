from mmengine.config import Config
import dict_utils 
from copy import deepcopy
import os

class ConfigDictGenerator:
    def __init__(self) -> None:
        self.config_bases = dict_utils.config_bases
        self.files_lookup_dict = dict_utils.method_files
    
    
    def generate_config_build_data_list(self, args) -> list:
        config_data = []
        for base_name, settings in self.config_bases.items():
            if base_name not in args.models:
                continue
            for algorithm_name in settings["algorithm_names"]:
                for backbone_name in settings["backbones"]:
                    
                    method_name = f'{algorithm_name}_{backbone_name}'
                    if method_name not in self.files_lookup_dict.keys():
                        continue
                    method_data = self.files_lookup_dict[method_name]
                    
                    for train_setting in settings["train_settings"]:
                        
                        train_setting = self.process_train_setting(
                                            args=args, train_setting=train_setting,
                                            method_data=method_data
                                        )
                        
                        for dataset_name in settings["datasets"]:
                            crop_sizes = args.crop_sizes
                            if crop_sizes is None:
                                crop_sizes = settings["crops"]
                            for crop_size in crop_sizes:
                                for iters in train_setting["iterations"]:
                                    
                                    
                                    checkpoints = []
                                    if args.scratch:
                                        checkpoints.append(None)
                                    if args.pretrained:
                                        for checkpoint in method_data["checkpoints"]:    
                                            checkpoints.append(checkpoint)
                                    for checkpoint in checkpoints:
                                        train_setting_iter = deepcopy(train_setting)
                                        train_setting_iter["iterations"] = iters
                                        train_setting_iter["checkpoint"] = checkpoint
                                        elements_dict = {
                                            "algorithm_name"    :   algorithm_name,
                                            "backbone_name"     :   backbone_name,
                                            "train_setting"     :   train_setting_iter,
                                            "dataset_name"      :   dataset_name,
                                            "crop_size"         :   crop_size
                                        }
                                        cfg_name = self.gen_config_name(elements_dict=elements_dict)
                                        # TODO workdirs is still hardcoded
                                        if args.unique and cfg_name in os.listdir("work_dirs"):
                                            continue
                                        cfg_build_data = {
                                            "cfg_name"          :       cfg_name,
                                            "base_cfg_path"     :       method_data["base_file_path"],
                                            "pretrained"        :       checkpoint is not None,
                                            "checkpoint"        :       checkpoint,
                                            "save_best"         :       args.save_best,
                                            "save_interval"     :       args.save_interval,
                                            "val_interval"      :       args.val_interval,
                                            "batch_size"        :       train_setting["batch_size"],
                                            "crop_size"         :       crop_size,
                                            "iterations"        :       iters,
                                            "verbose"           :       args.verbose,
                                            "configs"           :       args.config     
                                        }
                                        config_data.append(cfg_build_data)
        return config_data                         
    
    def gen_config_name(self, elements_dict: dict) -> str:
        # create easier alias for linewidth
        ed = elements_dict
        train_settings_str = self.gen_train_settings_str(ed["train_setting"])
        crop_str = f'{ed["crop_size"][0]}x{ed["crop_size"][1]}'
        return f'{ed["algorithm_name"]}_{ed["backbone_name"]}_{train_settings_str}_{ed["dataset_name"]}-{crop_str}'
    
    def gen_train_settings_str(self, train_setting: dict) -> str:
        iters = int(train_setting["iterations"])
        iter_str = str(iters)
        if iters >= 1000:
            iters /= 1000
            iter_str = f'{int(iters)}k'
        train_str = f'{train_setting["n_gpus"]}xb{train_setting["batch_size"]}'
        if train_setting["checkpoint"] is not None:
            train_str = f'{train_str}-pre-{train_setting["checkpoint"]["dataset_name"]}'
        return f'{train_str}-{iter_str}'
        
    def process_train_setting(self, args, train_setting: dict, method_data: dict) -> dict:
        train_setting_ = deepcopy(train_setting)
        if args.batch_size:
            train_setting_["batch_size"] = args.batch_size
        if args.iterations:
            train_setting_["iterations"] = args.iterations
        train_setting_["method_data"] = method_data
        return train_setting_
    

    def get_all_configs(self, args) -> list:
        cfg_build_data_list = self.generate_config_build_data_list(args=args)
        cfg_list = []
        for cfg_build_data in cfg_build_data_list:
            cfg_list.append(self.generate_config_from_build_data(cfg_build_data=cfg_build_data))
        return cfg_list
    

    def generate_config_from_build_data(self, cfg_build_data) -> Config:
        ConfigDictGenerator._generate_config_from_build_data(cfg_build_data=cfg_build_data)
    
    @staticmethod
    def _generate_config_from_build_data(cfg_build_data) -> Config:
        base_cfg = Config.fromfile(filename=cfg_build_data["base_cfg_path"])
        new_cfg_dict = base_cfg.to_dict()
        new_cfg = Config(cfg_dict=new_cfg_dict) #, filename=cfg_build_data["cfg_name"]
        additional_configs = []
        
        param_scheduler = new_cfg["param_scheduler"]
        
        param_scheduler = ConfigDictGenerator._update_param_scheduler(
                            param_scheduler=param_scheduler, iters=cfg_build_data["iterations"]
                        )
        additional_configs.append(
            dict(
                train_cfg = dict(
                    max_iters = cfg_build_data["iterations"]
                )
            )
        )
        if cfg_build_data["pretrained"] and cfg_build_data["checkpoint"]["path"]:
            additional_configs.append(
                dict(
                    load_from = cfg_build_data["checkpoint"]["path"]
                )
            )
        if cfg_build_data["save_best"]:
            additional_configs.append(
                dict(
                    default_hooks = dict(
                        checkpoint = dict(                                    
                            save_best='mIoU'
                        )
                    )
                )
            )
        
        if cfg_build_data["save_interval"]:
            additional_configs.append(
                dict(
                    default_hooks = dict(
                                        checkpoint=dict(                                    
                                                        interval=cfg_build_data["save_interval"]
                                                        )
                                        )
                            )
            )
        if cfg_build_data["val_interval"]:
            additional_configs.append(
                dict(
                    train_cfg = dict(
                                    val_interval=cfg_build_data["val_interval"]
                                    )
                )
            )
            
        if cfg_build_data["batch_size"]:
            additional_configs.append(
                dict(
                    train_dataloader = dict(
                        batch_size=cfg_build_data["batch_size"]
                    )
                )
            )
        if cfg_build_data["crop_size"]:
            crop_size = cfg_build_data["crop_size"] 
            data_preprocessor = dict(size=crop_size)
            # Load existing trainpipeline and change
            train_pipeline = new_cfg["train_pipeline"]
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
            new_cfg.merge_from_dict(additional_cfg)
        
        return new_cfg
    
    # TODO very simple now
    @staticmethod
    def _update_param_scheduler(param_scheduler, iters):
        param_scheduler[-1]["end"] = iters
    
    @staticmethod
    def _get_cfg_build_data(
                                cfg_name: str, base_cfg_path: str, pretrained: bool, 
                                checkpoint: dict, save_best: bool, save_interval: int,
                                val_interval: int, batch_size: int, crop_size: int,
                                iterations: int, verbose: bool = False, configs: list = {}
                            )-> dict:
        return  {
                    "cfg_name"          :       cfg_name,
                    "base_cfg_path"     :       base_cfg_path,
                    "pretrained"        :       pretrained,
                    "checkpoint"        :       checkpoint,
                    "save_best"         :       save_best,
                    "save_interval"     :       save_interval,
                    "val_interval"      :       val_interval,
                    "batch_size"        :       batch_size,
                    "crop_size"         :       crop_size,
                    "iterations"        :       iterations,
                    "verbose"           :       verbose,
                    "configs"           :       configs     
                }                
    
    def generate_config_names_list(self, args) -> list:
        
        return [build_item["cfg_name"] for build_item in self.generate_config_build_data_list(args=args)]
    
    
    def create_new_config_with_args(self, cfg_name, base_cfg, args):
        new_cfg_dict = base_cfg.to_dict()
        new_cfg = Config(cfg_dict=new_cfg_dict, filename=cfg_name)
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
            train_pipeline = new_cfg["train_pipeline"]
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
            new_cfg.merge_from_dict(additional_cfg)
        
        return new_cfg
        