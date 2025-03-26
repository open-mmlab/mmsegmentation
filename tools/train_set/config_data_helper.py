import yaml
import os
from pathlib import Path
from typing import Union
from dict_utils import DEFAULT_CONFIG_ROOT_PATH
import re
import dict_utils

class ConfigDataHelper:
    """_summary_
    reads and handles yaml metafiles
    """

    @staticmethod
    def _generate_available_project_list(
        config_root_path: Union[str, Path] = DEFAULT_CONFIG_ROOT_PATH
    ) -> list: 
        """generates list of available projects

        Args:
            config_root_path (Union[str, Path], optional):
                Defaults to DEFAULT_CONFIG_ROOT_PATH.

        Returns:
            list: list of available projects
        """
        
        return [
            project for project in os.listdir(config_root_path) 
                if not project == "_base_"
        ]
    
    # take out all irrelevant fields   
    @staticmethod
    def _extract_model_list(metafile_dict: dict) -> list:
        
        def _extract_method_name(cfg_name):
            idx = cfg_name.find("xb")
            return cfg_name[:idx - 2]
        
        def _extract_crop_size(cfg_name):
            
            split = re.split("_|-", cfg_name)
            crops = [crop for crop in split 
                     if len(crop.split('x')) == 2 
                     and crop.split('x')[0].isdigit() 
                     and crop.split('x')[1].isdigit()]
            if not crops:
                return None
            crop_str = crops[0]
            return (int(crop_str.split('x')[0]), int(crop_str.split('x')[1]))

        model_list = []
        if "Models" not in metafile_dict.keys():
            return []
        for model_dict in metafile_dict["Models"]:
            method_name = _extract_method_name(
                cfg_name=model_dict["Name"]
            )
            crop_size = _extract_crop_size(
                cfg_name=model_dict["Name"]
            )
            if "mIoU" in model_dict["Results"]["Metrics"].keys():
                mIoU = model_dict["Results"]["Metrics"]["mIoU"]
            else:
                # print(
                #     f'{model_dict["Name"]} : no MIoU, using {model_dict["Results"]["Metrics"]} instead'
                # )
                mIoU = list(model_dict["Results"]["Metrics"])[0][1]
            model_list.append(
            {
                "name"              :       model_dict["Name"],
                "cfg_path"          :       model_dict["Config"],
                "train_data"        :       model_dict["Metadata"]["Training Data"],
                "method"            :       method_name,
                "crop_size"         :       crop_size,
                "checkpoint_path"   :       model_dict["Weights"],
                "mIoU"              :       mIoU
            }
            )
        return model_list
    
    @staticmethod
    def _group_models_by_method(model_list: list) -> list:
        
        def _method_of_model_in_list(
            method_list: list, model_dict: dict
        ) -> bool:
            for method_dict in method_list:
                if _method_equal(
                    method_dict=method_dict, model_dict=model_dict
                ):
                    return True
            return False
        
        def _method_equal(method_dict: dict, model_dict: dict) -> bool:
            return method_dict["method"] == model_dict["method"]
        
        def _model_dict_equal(model_dict1: dict, model_dict2: dict) -> bool:
            method_eq = model_dict1["method"] == model_dict2["method"]
            data_eq = model_dict1["train_data"] == model_dict2["train_data"]
            return method_eq and data_eq
        
        def _find_model_in_method_dict(
            model_dict: dict, method_dict: dict
        ) -> dict:
            for model_dict_ in method_dict["models"]:
                if _model_dict_equal(
                    model_dict1=model_dict, 
                    model_dict2=model_dict_
                ):
                    return model_dict_
                
            return {}
        
        
        # best mIoU for now
        def _get_best_variant(model_dict1: dict, model_dict2: dict) -> dict:
            if model_dict1["mIoU"] > model_dict2["mIoU"]:
                return model_dict1
            return model_dict2
        
        def _add_new_model_to_method_dict(
            model_dict: dict, method_dict: dict
        ) -> None:
            method_dict["models"].append(model_dict)
        
        def _replace_model_in_method_dict(
            new_model_dict: dict, old_model_dict: dict,
            method_dict: dict
        ) -> None:
            idx = method_dict["models"].index(old_model_dict)
            method_dict["models"][idx] = new_model_dict
        
        def _generate_new_method_dict(model_dict: dict) -> dict:
            return {
                "method"        :   model_dict["method"],
                "models"        :   [model_dict]
            }
        
        method_list = []
        for model_dict in model_list:
        
            
            if not _method_of_model_in_list(
                method_list=method_list,
                model_dict=model_dict
            ):
                new_method_dict = _generate_new_method_dict(
                    model_dict=model_dict
                )
                method_list.append(new_method_dict)
                continue
            for method_dict in method_list:
                # method found in preexisting list
                if _method_equal(
                    method_dict=method_dict, model_dict=model_dict
                ):
                    
                    model_ret = _find_model_in_method_dict(
                        model_dict=model_dict,
                        method_dict=method_dict
                    )
                    
                    # if there is already such a model
                    if model_ret:
                        best_model = _get_best_variant(
                            model_dict1=model_dict,
                            model_dict2=model_ret
                        )
                       
                        _replace_model_in_method_dict(
                            new_model_dict=best_model,
                            old_model_dict=model_ret,
                            method_dict=method_dict
                        )
                    else:
                        _add_new_model_to_method_dict(
                            model_dict=model_dict,
                            method_dict=method_dict
                        )
        return method_list        
    @staticmethod
    def _method_list_trimmed_by_best_train_data(
        method_list: list
    ):
        def get_best_option(model_dict1, model_dict2):
            ranking = dict_utils.TrimData.accepted_dataset_list
            dataset1 = model_dict1["train_data"]
            dataset2 = model_dict2["train_data"]
            if ranking.index(dataset1) <= ranking.index(dataset2):
                return model_dict1
            return model_dict2
            
        for method_dict in method_list:
            if not method_dict["models"]:
                continue
            best_model_dict = method_dict["models"][0]
            for model_dict in method_dict["models"][1:]:
                best_model_dict = get_best_option(best_model_dict, model_dict)
            
            method_dict["models"] = [best_model_dict]
        return method_list                 
    @staticmethod
    def _method_list_trimmed_by_model_names(
        method_list: list, excluded_names: list
    ) -> list:
        for method_dict in method_list:
            new_model_list = []
            for model_dict in method_dict["models"]:
                if not model_dict["name"] in excluded_names:
                    new_model_list.append(model_dict)
                
            method_dict["models"] = new_model_list
        return method_list     
            
    @staticmethod
    def _method_list_trimmed_by_train_data(
        method_list: list, excluded_datasets: list
    ) -> list:
        for method_dict in method_list:
            new_model_list = []
            for model_dict in method_dict["models"]:
                if not model_dict["train_data"] in excluded_datasets:
                    new_model_list.append(model_dict)
            method_dict["models"] = new_model_list
        return method_list                  
    
    @staticmethod
    def _method_list_trimmed_by_method_names(
        method_list: list, excluded_method_names: list
    ) -> list:
        new_method_list = []
        for method_dict in method_list:
            if not method_dict["method"] in excluded_method_names:
                new_method_list.append(method_dict)
        return new_method_list

    
        
    @staticmethod
    def _read_metafile(
        project_name: str, 
        config_root_path: Union[str, Path] = DEFAULT_CONFIG_ROOT_PATH
    ) -> dict:
        """reads metafile.yaml in a safe way

        Args:
            project_name (str): 
                name of project to which metafile belongs
            config_root_path (Union[str, Path], optional): 
                Defaults to DEFAULT_CONFIG_ROOT_PATH.

        Returns:
            dict: contents of metafile
        """
        path = os.path.join(config_root_path, project_name, "metafile.yaml")
        if not os.path.exists(path):
            print(f'skipping load: path does not exist: {path}')
            return {}
        with open(path, "r") as f:
            try:
                metafile_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
                return {}
        return metafile_dict
    
    @staticmethod
    def _get_train_data_in_method_list(method_list: list) -> list:
        data_list = []
        for method_dict in method_list:
            for model_dict in method_dict["models"]:
                if not model_dict["train_data"] in data_list:
                    data_list.append(model_dict["train_data"])
        return data_list