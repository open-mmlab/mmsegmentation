from config_data_helper import ConfigDataHelper as CDHelper
from dict_utils import TrimData, dataset_info
from config_build_data import ConfigBuildData
from yaml import YAMLError
import os 

class ArgumentHandler:
    
    @staticmethod
    def _generate_config_build_data_list(args) -> list:
        config_build_data_list = []
        config_names = []
        method_list = ArgumentHandler._get_method_list_from_args(args=args)
        for method_dict in method_list:
            method_name = method_dict["method"]
            for model_dict in method_dict["models"]:
                # if [invalid_entry for key, invalid_entry 
                #     in model_dict.items() if invalid_entry is None]:
                #     continue
                checkpoint_paths = ArgumentHandler._get_checkpoint_paths(
                    args=args, model_dict=model_dict
                )
                for checkpoint_path in checkpoint_paths:
                    cfg_name = ArgumentHandler._generate_config_name(
                        args=args, 
                        method_name=method_name, 
                        pretrained=bool(checkpoint_path),
                        pretrain_data=model_dict["train_data"]
                    )
                    if cfg_name in config_names:
                        continue
                    if args.unique and cfg_name in os.listdir("work_dirs"):
                        continue
                    config_build_data = ConfigBuildData._get_cfg_build_data(
                        cfg_name=cfg_name, 
                        base_cfg_path=model_dict["cfg_path"],
                        dataset_cfg_path=dataset_info[args.dataset]["cfg_path"],
                        num_classes=dataset_info[args.dataset]["num_classes"],
                        pretrained=bool(checkpoint_path),
                        checkpoint_path=model_dict["checkpoint_path"],
                        pretrain_dataset=model_dict["train_data"],
                        save_best=args.save_best, 
                        save_interval=args.save_interval,
                        val_interval=args.val_interval, 
                        batch_size=args.batch_size,
                        crop_size=args.crop_size,
                        iterations=args.iterations,
                        epochs=args.epochs,
                        dataset_name=args.dataset
                    )
                    config_build_data_list.append(config_build_data)
                    config_names.append(config_build_data["cfg_name"])
        
            
        return config_build_data_list
    
    
    
    @staticmethod
    def _get_method_list_from_args(args) -> list:
        project_list = args.projects
        method_list = []
        for project_name in project_list:
            
            metafile_dict = CDHelper._read_metafile(project_name=project_name)
            if not metafile_dict:
                continue
            model_list = CDHelper._extract_model_list(
                metafile_dict=metafile_dict
            )
            method_list_ = CDHelper._group_models_by_method(
                model_list=model_list
            )
            if args.trim_method_list:
                (accepted_dataset_list, excluded_method_names) = TrimData._get_exclusion_lists_from_args(args=args)
                train_data_list = CDHelper._get_train_data_in_method_list(method_list=method_list_)
                train_data_list_exclude = [train_data for train_data in train_data_list 
                        if train_data not in accepted_dataset_list]

                method_list_ = CDHelper._method_list_trimmed_by_train_data(
                    method_list=method_list_,
                    excluded_datasets=train_data_list_exclude
                )
                method_list_ = CDHelper._method_list_trimmed_by_best_train_data(
                    method_list=method_list_ 
                )
                
                method_list_ = CDHelper._method_list_trimmed_by_method_names(
                    method_list=method_list_,
                    excluded_method_names=excluded_method_names
                )
                
                
            method_list += method_list_
        return method_list
    
    @staticmethod        
    def _generate_config_name(
        args, method_name: str, 
        pretrained: bool, pretrain_data:  str
    ) -> str:
        
        
        crop_size = args.crop_size
        crop_str = f'{crop_size[0]}x{crop_size[1]}'
        
        dataset_str = f'{args.dataset.lower().replace(" ", "")}-{crop_str}'
        train_settings_str = ArgumentHandler._generate_train_settings_str(
            args=args, pretrained=pretrained, pretrain_data=pretrain_data
        )
        return f'{method_name}_{train_settings_str}_{dataset_str}'  
    
    @staticmethod      
    def _generate_train_settings_str(
        args, pretrained: bool, pretrain_data:  str
    ) -> str:
        if args.iterations:
            iters = int(args.iterations)
            iter_str = str(iters)
            if iters >= 1000:
                iters /= 1000
                iter_str = f'{int(iters)}k'
        if args.epochs:
            iter_str = f'{int(args.epochs)}epochs'
        train_str = f'{args.n_gpus}xb{args.batch_size}'
        if pretrained:
            train_str += f'-pre-{pretrain_data.lower().replace(" ", "")}'
        return f'{train_str}-{iter_str}'
    
    @staticmethod
    def _get_checkpoint_paths(args, model_dict: dict) -> list:
        checkpoint_paths = []
        if args.scratch:
            checkpoint_paths.append(None)
        if args.pretrained:
            checkpoint_paths.append(model_dict["checkpoint_path"])
        return checkpoint_paths