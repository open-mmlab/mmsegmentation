import copy
from mmengine.config import Config, ConfigDict
import dict_utils 
from copy import deepcopy
import os


class ConfigDictGenerator:
    def __init__(self) -> None:
        self.config_bases = dict_utils.config_bases
        self.files_lookup_dict = dict_utils.method_files
        self.dataset_info = dict_utils.dataset_info
        self.irrelevant_arg_names = [
            "models", "checkpoint", 
            "config", "verbose"
        ]
    
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
                    checkpoints = self.get_checkpoints(args, method_data)
                    for checkpoint in checkpoints:
                        train_dict = self.generate_train_dict_from_args(
                            args=args, checkpoint=checkpoint
                        )
                        
                        cfg_name = self.generate_config_name(
                            train_dict=train_dict, method_name=method_name
                        )
                        if args.unique and cfg_name in os.listdir("work_dirs"):
                            continue
                        
                        cfg_build_data = self.cfg_build_data_from_train_dict(
                            train_dict=train_dict, cfg_name=cfg_name,
                            base_cfg_path=method_data["base_file_path"]
                        )
                        config_data.append(cfg_build_data)
        return config_data                         
    
    def generate_train_dict_from_args(self, args, checkpoint: dict) -> dict:
        train_dict = dict()
        for arg_name, val in args.__dict__.items():
            if arg_name in self.irrelevant_arg_names:
                continue
            train_dict[arg_name] = val
        train_dict["checkpoint"] = checkpoint
        train_dict["pretrained"] = checkpoint is not None
        return train_dict
    


    def generate_config_name(self, train_dict: dict, method_name: str) -> str:
        train_settings_str = self.generate_train_settings_str(
            train_dict=train_dict
        )
        crop_size = train_dict["crop_size"]
        crop_str = f'{crop_size[0]}x{crop_size[1]}'
        
        dataset_str = f'{train_dict["dataset"]}-{crop_str}'
        
        return f'{method_name}_{train_settings_str}_{dataset_str}'
    
    def generate_train_settings_str(self, train_dict: dict) -> str:
        if train_dict["iterations"]:
            iters = int(train_dict["iterations"])
            iter_str = str(iters)
            if iters >= 1000:
                iters /= 1000
                iter_str = f'{int(iters)}k'
        if train_dict["epochs"]:
            iter_str = f'{int(train_dict["epochs"])}epochs'
        train_str = f'{train_dict["n_gpus"]}xb{train_dict["batch_size"]}'
        if train_dict["checkpoint"] is not None:
            train_str += f'-pre-{train_dict["checkpoint"]["dataset_name"]}'
        return f'{train_str}-{iter_str}'
        
    def cfg_build_data_from_train_dict(
        self, train_dict: dict, 
        cfg_name: str, base_cfg_path: str
    ):
        cfg_build_data = copy.deepcopy(train_dict)
        cfg_build_data["cfg_name"] = cfg_name
        cfg_build_data["base_cfg_path"] = base_cfg_path
        return cfg_build_data

    def get_all_configs(self, args) -> list:
        cfg_build_data_list = self.generate_config_build_data_list(args=args)
        cfg_list = []
        for cfg_build_data in cfg_build_data_list:
            cfg_list.append(
                self.generate_config_from_build_data(
                    cfg_build_data=cfg_build_data
                )
            )
        return cfg_list
    

    def generate_config_from_build_data(self, cfg_build_data) -> Config:
        ConfigDictGenerator._generate_config_from_build_data(
            cfg_build_data=cfg_build_data
        )
    
    @staticmethod
    def _generate_config_from_build_data(cfg_build_data: dict) -> Config:
        base_cfg = Config.fromfile(filename=cfg_build_data["base_cfg_path"])
        new_cfg_dict = base_cfg.to_dict()
        new_cfg = Config(cfg_dict=new_cfg_dict) 
       
        additional_configs = []
        
        ConfigDictGenerator.apply_dataset(
            cfg_build_data=cfg_build_data,
            new_cfg=new_cfg
        )
        
        if cfg_build_data["iterations"]:
            param_scheduler = new_cfg["param_scheduler"]
            # update param scheduler
            iteration = 0
            iterations_per_step = int(
                cfg_build_data["iterations"] // len(param_scheduler)
            )
            for schedule in param_scheduler:
                schedule["begin"] = iteration
                schedule["end"] = iteration + iterations_per_step
                iteration += iterations_per_step
            param_scheduler[-1]["end"] = cfg_build_data["iterations"]
            # # TODO temp
            # param_scheduler = [
            #     dict(
            #         begin=0, by_epoch=False, end=cfg_build_data["iterations"],
            #         start_factor=1.0, end_factor = 0.01,
            #         type='LinearLR'
            #     )
            # ]
            # new_cfg["param_scheduler"] = param_scheduler
            
            
            additional_configs.append(
                dict(
                    train_cfg = dict(
                        max_iters = cfg_build_data["iterations"]
                    )
                )
            )
        if cfg_build_data["epochs"]:
            # update param scheduler
            param_scheduler = new_cfg["param_scheduler"]
            epoch = 0
            epoch_per_step = int(
                cfg_build_data["epochs"] // len(param_scheduler)
            )
            for schedule in param_scheduler:
                schedule["begin"] = epoch
                schedule["end"] = epoch + epoch_per_step
                schedule["by_epoch"] = True
                epoch += epoch_per_step
            param_scheduler[-1]["end"] = cfg_build_data["epochs"]
                
            
            additional_configs.append(
                dict(
                    train_dataloader = dict(
                        sampler = dict(
                            type = "DefaultSampler",
                            shuffle = True
                        ),
                        drop_last = True
                    )
                )
            )
            additional_configs.append(
                dict(
                    train_cfg = dict(
                        max_epochs = cfg_build_data["epochs"],
                        type = "EpochBasedTrainLoop"
                    )
                )
            )
            
            train_cfg = dict(
                max_epochs = cfg_build_data["epochs"],
                type = "EpochBasedTrainLoop"
            )
            
            new_cfg["train_cfg"] = train_cfg     
            
            additional_configs.append(
                dict(
                    default_hooks = dict(
                        checkpoint = dict(
                            by_epoch = True
                        ), 
                        logger = dict(
                            log_metric_by_epoch = True
                        )
                    )
                )
            )
            
            additional_configs.append(
                dict(
                    log_processor = dict(
                        by_epoch = True
                    )
                )
            )
            
            
            
        if cfg_build_data["pretrained"] and cfg_build_data["checkpoint_path"]:
            additional_configs.append(
                dict(
                    load_from = cfg_build_data["checkpoint_path"]
                )
            )
        if cfg_build_data["save_best"]:
            additional_configs.append(
                dict(
                    default_hooks = dict(
                        checkpoint = dict(                                    
                            save_best = 'mIoU'
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
            ConfigDictGenerator._apply_crop_size(
                additional_configs=additional_configs,
                cfg_build_data=cfg_build_data,
                new_cfg=new_cfg
            )
                
        for additional_cfg in additional_configs:
            new_cfg.merge_from_dict(additional_cfg)
        
        return new_cfg
    
    
        
    
    @staticmethod
    def _get_cfg_build_data(
        cfg_name: str, base_cfg_path: str, pretrained: bool, 
        checkpoint: dict, save_best: bool, save_interval: int,
        val_interval: int, batch_size: int, crop_size: int,
        iterations: int, epochs: int, dataset_name: str
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
                    "epochs"            :       epochs,
                    "dataset"           :       dataset_name    
                }                
    
    @staticmethod
    def _apply_crop_size(
        additional_configs: list, 
        cfg_build_data: dict, new_cfg: Config
    ):
        crop_size = cfg_build_data["crop_size"] 
        data_preprocessor = dict(size=crop_size)
        # Load existing trainpipeline and change
        train_pipeline = new_cfg["train_pipeline"]
        previous_crop_size = None
        for step_dict in train_pipeline:
            if step_dict["type"] == "RandomCrop":
                previous_crop_size = step_dict["crop_size"]
                step_dict["crop_size"] = crop_size
            if step_dict["type"] == "RandomResize":
                step_dict["scale"] = (2048, min(crop_size))
            if step_dict["type"] == "RandomChoiceResize":
                step_dict["scales"] = [int(min(crop_size) * x * 0.1) for x in range(5, 21)]
                step_dict["max_size"] = 2048
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
        
        test_pipeline = new_cfg["test_pipeline"]
        for step_dict in test_pipeline:
            if step_dict["type"] == "Resize":
                step_dict["scale"] = (2048, min(crop_size))
            if step_dict["type"] == "RandomCrop":
                step_dict["crop_size"] = crop_size
            if step_dict["type"] == "RandomResize":
                step_dict["scale"] = (2048, min(crop_size))
            if step_dict["type"] == "RandomChoiceResize":
                step_dict["scales"] = [int(min(crop_size) * x * 0.1) for x in range(5, 21)]
                step_dict["max_size"] = 2048
        additional_configs.append(
            dict(
                test_pipeline=test_pipeline
            )
        )
        additional_configs.append(
            dict(
                test_dataloader = dict(
                    dataset = dict(
                        pipeline=test_pipeline
                    )
                )
            )
        )
        
        additional_configs.append(
            dict(
                val_pipeline=test_pipeline
            )
        )
        additional_configs.append(
            dict(
                val_dataloader = dict(
                    dataset = dict(
                        pipeline=test_pipeline
                    )
                )
            )
        )
        
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
        
        # Only few models e.g. convnext have need for this
        
        if "test_cfg" in new_cfg["model"].keys():
            if "crop_size" in new_cfg["model"]["test_cfg"]:
                new_cfg["model"]["test_cfg"]["crop_size"] = crop_size

        
        # TODO hardcoded now but with  search can be generalized
        if "backbone" in new_cfg["model"].keys():
            if "img_size" in new_cfg["model"]["backbone"].keys():
                new_cfg["model"]["backbone"]["img_size"] = crop_size
        if "image_encoder" in new_cfg["model"].keys():
            if "img_size" in new_cfg["model"]["image_encoder"].keys():
                new_cfg["model"]["image_encoder"]["img_size"] = crop_size 
        
        if cfg_build_data["pretrained"] and previous_crop_size:
            if "backbone" in new_cfg["model"].keys():
                if "pretrain_img_size" in new_cfg["model"]["backbone"].keys():
                    new_cfg["model"]["backbone"]["pretrain_img_size"] = previous_crop_size
            if "image_encoder" in new_cfg["model"].keys():
                if "img_size" in new_cfg["model"]["image_encoder"].keys():
                    new_cfg["model"]["image_encoder"]["pretrain_img_size"] = previous_crop_size 
  
    # TODO maybe preserve specific train_pipelines 
    @staticmethod
    def apply_dataset(cfg_build_data: dict, new_cfg: Config):
        dataset_info = dict_utils.dataset_info[cfg_build_data["dataset"]]
        num_classes = dataset_info["num_classes"]
        class_weight = dataset_info["class_weight"]
        if "mask" in cfg_build_data["cfg_name"] and "former" in cfg_build_data["cfg_name"]:
            class_weight = [0.1] + ([1.0] * num_classes)
        dataset_cfg = Config.fromfile(
            dataset_info["cfg_path"]
        )
        for key, value in dataset_cfg.items():
            new_cfg[key] = value
        dict_utils.BFS_change_key(
            cfg=new_cfg, 
            target_key="num_classes", 
            new_value=num_classes
        )
        dict_utils.BFS_change_key(
            cfg=new_cfg,
            target_key="class_weight",
            new_value=class_weight
        )
        # dict_utils.BFS_change_key(
        #     cfg=new_cfg,
        #     target_key="size_divisor",
        #     new_value=None
        # )
        
        # for component_name, component in new_cfg["model"].items():
        #     if component_name == "num_classes":
        #         component = num_classes
        #     if type(component) is ConfigDict:
        #         if "num_classes" in component.keys():
        #             component["num_classes"] = num_classes
        #     if type(component) is list:
        #         for element in component:
        #              if type(element) is ConfigDict:
        #                 if "num_classes" in element.keys():
        #                     element["num_classes"] = num_classes
               
        
        
           
    
    def generate_config_names_list(self, args) -> list:
        
        return [build_item["cfg_name"] for build_item in self.generate_config_build_data_list(args=args)]
    
    
    