from config_data_helper import ConfigDataHelper as CDH

import dict_utils
import os
from mmengine import Config


config_root_path = "../../configs/"

cfg = Config.fromfile(os.path.join(config_root_path, "_base_", "models", "cgnet.py"))

print(cfg.dump())

dict_utils.BFS_change_key(cfg=cfg, target_key="class_weight", new_value=[])

print('#' * 100)
print(cfg.dump())
# project = "pspnet"

# project_dict = CDH._read_metafile(project_name=project, config_root_path=config_root_path)

# model_list = CDH._extract_model_list(metafile_dict=project_dict)

# # for model in model_list:
# #     print('-' * 40)
# #     for key, val in model.items():
# #         print(f'{key} : {val}')

# # print('#' * 40)
# method_list = CDH._group_models_by_method(model_list=model_list)

# # for method in method_list:
    
# #     print(method["method"])
# #     print('#' * 80)
# #     for model in method["models"]:
# #         print('-' * 40)
# #         for key, val in model.items():
# #             print(f'{key} : {val}\n') 

# # model_names = ["pspnet_r50-d8_4xb2-80k_cityscapes-769x769", "pspnet_r50-d8_4xb4-160k_ade20k-512x512"]
# # method_list = CDH._method_list_trimmed_by_model_names(method_list=method_list, excluded_names=model_names)

# # print('%' * 80)
# # for method in method_list:
    
# #     print(method["method"])
# #     print('#' * 80)
# #     for model in method["models"]:
# #         print('-' * 40)
# #         for key, val in model.items():
# #             print(f'{key} : {val}\n') 

# train_data_list = CDH._get_train_data_in_method_list(method_list=method_list)
# train_data_list_exclude = [train_data for train_data in train_data_list 
#                         if train_data not in dict_utils.accepted_dataset_list]

# excluded_method_names = ["pspnet_r50-d8"]
# method_list = CDH._method_list_trimmed_by_method_names(
#     method_list=method_list,
#     excluded_method_names=excluded_method_names    
# )
# print('%' * 80)
# for method in method_list:
    
#     print(method["method"])
#     print('#' * 80)
#     for model in method["models"]:
#         print('-' * 40)
#         for key, val in model.items():
#             print(f'{key} : {val}\n') 
