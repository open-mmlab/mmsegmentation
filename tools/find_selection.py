
from train_set.dict_utils import TrimData
import os
from mmengine import Config
from copy import copy, deepcopy
import markdown
import re
from mmengine.analysis.complexity_analysis import (
    parameter_count_table, 
    parameter_count
)
from mmengine.registry import init_default_scope

from mmseg.registry import MODELS

def parse_row(row_str):
    row_ = row_str.split("|")
    row = []
    for item in row_:
        if item == "" or item == "\n":
            continue
        row.append(item.strip())
    return row 
        
def is_table_row(line):
    if len(line) < 2:
        return False
    return line[0] == "|" and line[-1] == "|"

def is_first_row(row):
    return row[0] == "Method" and row[1] == "Backbone"

def is_sep_row(row):
    return "--" in row[0]

def extract_dataset_name(line):
    if len(line) < 4:
        return ""
    if not line[:3] == "###":
        return ""
    trimmed_line = line[4:]
    trimmed_line = trimmed_line[:-1]
    return trimmed_line

def same_model_diff_sched(model_dict0, model_dict1):
    if model_dict0 is model_dict1:
        return False
    if model_dict0["Method"] != model_dict1["Method"]:
        return False
    if model_dict0["Backbone"] != model_dict1["Backbone"]:
        return False
   
    if model_dict0["Crop Size"] != model_dict1["Crop Size"]:
        return False 
    
    if model_dict0["dataset"] != model_dict1["dataset"]:
        if (
            (
                model_dict0["dataset"] == "Pascal Context"
                and
                model_dict1["dataset"] == "Pascal Context 59"
            )
            or 
            (
                model_dict1["dataset"] == "Pascal Context"
                and
                model_dict0["dataset"] == "Pascal Context 59"
            )
        ):
            return True
        return False
    return True

def complete_model_dict(model_dict, model_dict_list):
    for model_dict_ in model_dict_list:
        if model_dict is model_dict_:
            continue
        if same_model_diff_sched(
            model_dict0=model_dict_,
            model_dict1=model_dict
        ):
            if model_dict["Mem (GB)"] == '-' and model_dict_["Mem (GB)"] != '-':
                model_dict["Mem (GB)"] = model_dict_["Mem (GB)"]
            if (
                    model_dict["Inf time (fps)"] == '-'
                    and 
                    model_dict_["Inf time (fps)"] != '-'
            ):
                model_dict["Inf time (fps)"] = model_dict_["Inf time (fps)"]
            return model_dict
    return None # TODO not sure just for test


def fill_in_missing_vals(
    model_dict_list,
    keys_to_fix = [
        "Mem (GB)",
        "Inf time (fps)",
        "mIoU"
    ]
):
    model_dict_list_ = []
    for model_dict in model_dict_list:
        if '-' in [
            value for key, value in model_dict.items() if key in keys_to_fix
        ]:
        # if '-' in model_dict.values():
            model_dict = complete_model_dict(
                model_dict=model_dict,
                model_dict_list=model_dict_list
            )
            if model_dict is None:
                continue
        model_dict_list_.append(model_dict)
    return model_dict_list_ 

def fix_dict_types(
    model_dict_list,
    keys_to_fix = [
        "Mem (GB)",
        "Inf time (fps)",
        "mIoU"
    ]
):
    model_dict_list_ = []
    for model_dict in model_dict_list:        
        for key in keys_to_fix:
            try:
                val = re.findall(r"[-+]?(?:\d*\.*\d+)", model_dict[key])[0]
                
            except:
                continue
            model_dict[key] = float(val)
        model_dict_list_.append(model_dict)
    return model_dict_list_

def dict_keys_valid(
    model_dict,
    essential_keys = [
        "Method",
        "Backbone",
        "Crop Size",
        "dataset",
        "Mem (GB)",
        "Inf time (fps)",
        "mIoU"
    ]
):
    for key in essential_keys:
        if key not in model_dict.keys():
            
            
            return False
    return True

def comparing_dict_vals_valid(
    model_dict,
    comparing_keys = [
        "Mem (GB)",
        "Inf time (fps)",
        "mIoU"
    ]
):
    for key in comparing_keys:
        if not isinstance(model_dict[key], (int, float)):
            return False
    return True   
def get_comparable_model_dict_list(
    model_dict_list,
    comparing_keys = [
        "Mem (GB)",
        "Inf time (fps)",
        "mIoU"
    ]
):
    model_dict_list_ = []
    for model_dict in model_dict_list:
        if not comparing_dict_vals_valid(
            model_dict=model_dict,
            comparing_keys=comparing_keys
        ):
            continue
        model_dict_list_.append(model_dict)
    
    return model_dict_list_

def remove_invalid_dicts(
    model_dict_list,
    essential_keys = [
        "Method",
        "Backbone",
        "Crop Size",
        "dataset",
        "Mem (GB)",
        "Inf time (fps)",
        "mIoU"
    ]
):
    model_dict_list_ = []
    for model_dict in model_dict_list:
        
        if model_dict is None:
            continue
        if not dict_keys_valid(
            model_dict=model_dict, essential_keys=essential_keys
        ):
            continue
            
        if not os.path.exists(model_dict["config"]):
            continue
        model_dict_list_.append(model_dict)
    return model_dict_list_

def is_valid_dataset_name(
    dataset_name,
    accepted_dataset_list = copy(TrimData.accepted_dataset_list)
):
    return dataset_name in accepted_dataset_list

def is_dataset_name(
    line,
    accepted_dataset_list = copy(TrimData.accepted_dataset_list)
):
    if len(line) < 4:
        return False
    if not line[:3] == "###":
        return False
    dataset_name = extract_dataset_name(line=line)
    if dataset_name == "":
        return False
    return True

def make_model_dict(row, data_set_name, labels):
    model_dict = {}
    for label in labels:
        if label == "download":
            continue
        if label == "config":
            config_path = row[labels.index(label)].split("main/")[-1][:-1]
            model_dict["config"] = config_path
            continue
        model_dict[label] = row[labels.index(label)] 
    model_dict["dataset"] = data_set_name
    return model_dict

def get_readme_data(
    project_path, 
    accepted_dataset_list = copy(TrimData.accepted_dataset_list)
):
    readme_path =  os.path.join(project_path, "README.md")
    if not os.path.exists(readme_path):
        return None
    model_dict_list = []
    current_dataset_name = ""
    labels = []
    
    with open(readme_path, 'r') as readme_file:
        for line in readme_file.readlines():
            if is_dataset_name(line=line):
                current_dataset_name = extract_dataset_name(line=line)
                
            if not is_valid_dataset_name(current_dataset_name):
                continue   

            if is_table_row(line=line[:-1]):
                row = parse_row(row_str=line)
                if is_sep_row(row=row):
                    continue
                
                if is_first_row(row=row):
                    labels = row
                else:
                    model_dict_list.append(make_model_dict(
                            row=row,
                            data_set_name=current_dataset_name,
                            labels=labels
                        )
                    )
    return model_dict_list
                
def postprocesses_model_dict_list(model_dict_list):
    model_dict_list = remove_invalid_dicts(model_dict_list=model_dict_list)
    if model_dict_list == []:
        return []
    # fill in missing values
    model_dict_list = fill_in_missing_vals(model_dict_list=model_dict_list)
   
    # fix types
    model_dict_list = fix_dict_types(model_dict_list=model_dict_list)
    
    # filter out last faulty data
    model_dict_list = get_comparable_model_dict_list(
        model_dict_list=model_dict_list
    )

    return model_dict_list

def get_all_model_dicts(
    project_list,
    config_root_path = "configs/",
    accepted_dataset_list = copy(TrimData.accepted_dataset_list)
):

    model_dict_list = []
    for project_name in project_list:
        project_path = os.path.join(config_root_path, project_name)
        project_model_dict_list = get_readme_data(
            project_path=project_path,
            accepted_dataset_list=accepted_dataset_list
        )
       
        if project_model_dict_list is None:
            continue
        
        project_model_dict_list = postprocesses_model_dict_list(
            model_dict_list=project_model_dict_list
        )
        model_dict_list += project_model_dict_list
    return model_dict_list



class ScoreCollection:
    
    calc_options = [
        "default",
        "fps_per_GB",
        "mIoU_per_spf"
    ]
    
    @staticmethod
    def get_score(
        model_dict, 
        method = "default",
        factors = {
            "mem_factor" : 1,
            "fps_factor" : 1,
            "mIoU_factor" : 1
        }
    ):
        if method == "fps_per_GB":
            return ScoreCollection.fps_per_GB(
                model_dict=model_dict,
                mem_factor=factors["mem_factor"],
                fps_factor=factors["fps_factor"]
            )
        if method == "mIoU_per_spf":
            return ScoreCollection.mIoU_per_spf(
                model_dict=model_dict
            )
        return ScoreCollection.default_calc(
            model_dict=model_dict,
            mem_factor=factors["mem_factor"],
            fps_factor=factors["fps_factor"],
            mIoU_factor=factors["mIoU_factor"]
        )
    @staticmethod
    def default_calc(
        model_dict, 
        mem_factor = 1,
        fps_factor = 1,
        mIoU_factor = 1
    ):
        num = (
            fps_factor * model_dict["Inf time (fps)"]
            +
            mIoU_factor * model_dict["mIoU"]
        )
        den = mem_factor * model_dict["Mem (GB)"]
        if den == 0:
            den = 1
        return num / den
    @staticmethod
    def fps_per_GB(
        model_dict,
        mem_factor = 1,
        fps_factor = 1
    ):
        num = fps_factor * model_dict["Inf time (fps)"]
        den = mem_factor * model_dict["Mem (GB)"]
        if den == 0:
            den = 1
        return num / den
    
    # inverse of fps
    @staticmethod
    def mIoU_per_spf(
        model_dict
    ):
        num = model_dict["mIoU"]
        den = 1.0 / model_dict["Inf time (fps)"]
        if den == 0:
            den = 1
        return num / den

def add_custom_score(
    model_dict_list,
    score_name = "custom_score", 
    calc_method = "default",
    factors = {
        "mem_factor" : 1,
        "fps_factor" : 1,
        "mIoU_factor" : 1
    }
):
    
       
    for model_dict in model_dict_list:
        model_dict[score_name] = ScoreCollection.get_score(
            model_dict=model_dict,
            method=calc_method,
            factors=factors
        )
    return model_dict_list

def add_n_params_to_model_dicts(model_dict_list):
    for model_dict in model_dict_list:
        cfg = Config.fromfile(model_dict["config"])
        init_default_scope(cfg.get('default_scope', 'mmseg'))
        cfg.model.train_cfg = None
        model = MODELS.build(cfg.model)
        model_dict["n_params"] = parameter_count(model=model)['']
    return model_dict_list
        
def sorted_by_metric(
    model_dict_list, 
    metric ='mIoU', 
    descending=True
):
    sort_list =  sorted(
        model_dict_list, key=lambda d : d[metric], reverse=descending
    ) 
    return sort_list


def print_trimmed_model_dict(
    model_dict
):
    config_path = model_dict["config"]
    print(f"config : {config_path}")
    for key, val in model_dict.items():
        if isinstance(val, (int, float)):
            print(f"{key} : {val}")   
               
config_root_path = "configs/"


project_list = [
            project for project in os.listdir(config_root_path) 
                if not project == "_base_"
]
for project_name in project_list:
    print(project_name)
model_dict_list = get_all_model_dicts(
    project_list=project_list
)

print(len(model_dict_list))



model_dict_list = add_custom_score(
    model_dict_list=model_dict_list,
    score_name="custom_score"
)

model_dict_list = add_custom_score(
    model_dict_list=model_dict_list,
    score_name="fps_per_GB",
    calc_method="fps_per_GB"
)

model_dict_list = add_custom_score(
    model_dict_list=model_dict_list,
    score_name="mIoU_per_spf",
    calc_method="mIoU_per_spf"
)

# One very specific case:
for model_dict in model_dict_list:
    if "." not in str(model_dict["Mem (GB)"])[:3]:  
        model_dict["Mem (GB)"] = model_dict["Mem (GB)"] / 1000.0


def get_batch_size_from_cfg_name(cfg_name):
    return int(
        cfg_name.split("xb")[-1].split("-")[0]
    )



# filter on memory
# TODO: not very good estimator
def remove_too_big_models(model_dict_list, max_mem_per_batch = 2.5):
    model_dict_list_ = []
    for model_dict in model_dict_list:
        cfg_name = model_dict["config"].split("/")[-1]
        batch_size = get_batch_size_from_cfg_name(
            cfg_name=cfg_name
        )
        if model_dict["Mem (GB)"] / float(batch_size) > max_mem_per_batch:
            continue
        model_dict_list_.append(
            model_dict
        )
    return model_dict_list_

def get_best_model(model_group, metric = "mIoU"):
    return sorted_by_metric(
        model_dict_list=model_group,
        metric=metric,
        descending=True
    )[0]
    

def get_unique_model_list(model_dict_list):
    unique_list = []
    for model_dict in model_dict_list:
        if len(
            [
                m_dict for m_dict in unique_list
                    if same_model_diff_sched(
                        model_dict0=model_dict,
                        model_dict1=m_dict
                    ) or model_dict["config"] == m_dict["config"]
            ]
        ) > 0:
            continue
        model_group = [model_dict]
        for model_dict1 in model_dict_list:
            if model_dict is model_dict1:
                continue
            if same_model_diff_sched(
                model_dict0=model_dict,
                model_dict1=model_dict1
            ):
                model_group.append(model_dict1)
        unique_list.append(
            get_best_model(model_group=model_group)
        )
    return unique_list
        
        
# model_dict_list = add_n_params_to_model_dicts(model_dict_list=model_dict_list)

def record_score(model_dict, score_list, rank, n_items, metric):
    
    rank_score = n_items - rank
    if metric == "Inf time (fps)" or metric == "mIoU":
        rank_score *= 2
    if model_dict["config"] not in [
        item["config"] for item in score_list 
    ]:
        model_dict = deepcopy(model_dict)
        model_dict["score"] = {
            "count"             :          1,
            "metric_results"    :          [
                    {
                        "rank"      :       rank,
                        "metric"    :       metric     
                    }
                ],
            "total_rank_score"  :           rank_score
        }
        score_list.append(
            model_dict
        )
        return score_list
    for item in score_list:
        if item["config"] == model_dict["config"]:
            item["score"]["count"] += 1
            item["score"]["metric_results"].append(
                {
                    "rank"      :       rank,
                    "metric"    :       metric
                }
            )
            item["score"]["total_rank_score"] += rank_score
            break
    return score_list



def get_sorted_by_metric_for_dataset(
    dataset_name,
    model_dict_list,
    metric_param_list
):
    
    model_dict_list_ = [
        model_dict for model_dict in model_dict_list 
            if model_dict["dataset"] == dataset_name
    ]
    sorted_by_metric_dict = {}
    for param in metric_param_list:
        
        sorted_dict_list = sorted_by_metric(
            model_dict_list=model_dict_list_,
            metric=param["metric"],
            descending=param["descending"]
        )
        sorted_by_metric_dict[param["metric"]] = sorted_dict_list
    return sorted_by_metric_dict

def get_top_n_metric_dict(
    sorted_by_metric_dict,
    n_items = 10
):  
    top_n_metric_dict = {}
    for metric, sorted_dict_list in sorted_by_metric_dict.items():
        top_n_metric_dict[metric] = sorted_dict_list[:n_items]
    return top_n_metric_dict

def print_n_top_items(top_n_metric_dict):
    for metric, top_n in top_n_metric_dict.items():
        print(f"\ntop {len(top_n)} of {metric}:\n")
        rank = 1
        for model_dict in top_n:
            print(f"Rank: {rank}")
            print_trimmed_model_dict(model_dict=model_dict)
            print()
            rank += 1

def get_score_list_dataset(top_n_metric_dict):
    score_list = []
    for metric, top_n in top_n_metric_dict.items():
        rank = 1
        for model_dict in top_n:
            score_list = record_score(
                model_dict=model_dict,
                score_list=score_list,
                rank=rank,
                n_items=n_items,
                metric=metric
            )
            rank += 1
    return score_list



def sorted_score(score_list, n_items = 10):
    sorted_score_list = sorted(
        score_list, 
        key=lambda d : d["score"]["total_rank_score"], 
        reverse=True
    ) 
    if len(sorted_score_list) >= n_items:      
        return sorted_score_list[:n_items]
    return sorted_score_list


def print_sorted_score(sorted_score_list):
    print(f"sorted scores ({len(sorted_score_list)}): ")
    for moded_model_dict in sorted_score_list:
        print_trimmed_model_dict(model_dict=moded_model_dict)
        print("scores: ")
        for key, val in moded_model_dict["score"].items():
            if key == "metric_results":
                sorted_rank = sorted(
                    val,
                    key=lambda item: item["rank"]
                )
                print(f"{key}:")
                for record in sorted_rank:
                    print(f" - rank : {record['rank']}, metric : {record['metric']}")
                continue
            print(f"{key} : {val}")
        print()



def write_selected_files(selected_model_dict_list):
    for model_dict in selected_model_dict_list:
        
        cfg_path = model_dict["config"]
        cfg_dir = cfg_path.split("/")[-2]
        lines = []
        with open(cfg_path, 'r') as cfg_file:
            lines = cfg_file.readlines()
        cfg_name = cfg_path.split("/")[-1]
        dst_path = os.path.join(
            "configs/my_configs/",
            cfg_name
        )
        with open(dst_path, 'w') as dest_file:
            for line in lines:
                if line[:len("_base_")] == "_base_":
                    final_part_line = line.split('=')[-1]
                    final_part_line = final_part_line.strip()
                    # multiple base files
                    if final_part_line == '[':
                        pass
                    if final_part_line[:len("'./")] == "'./":
                        base_name = final_part_line[len("'./"):-1]
                        base_path = os.path.join("../", cfg_dir, base_name)
                        dest_file.write(f"_base_ = '{base_path}'\n")
                        continue
                    if final_part_line[:len("['./")] == "['./":
                        base_name = final_part_line[len("['./"):-2]
                        base_path = os.path.join("../", cfg_dir, base_name)
                        dest_file.write(f"_base_ = ['{base_path}']\n")
                        continue
                dest_file.write(f"{line}")

def select_configs(selected_model_dict_list, provisonal_list):
    for provisional_model_dict in provisonal_list:
        if provisional_model_dict["config"] in [
            model_dict["config"] for model_dict in selected_model_dict_list
        ]:
            continue
        selected_model_dict_list.append(
            provisional_model_dict
        )
    return selected_model_dict_list


metric_param_list = [
    {
        "metric"        :       "custom_score",
        "descending"    :       True
    },
    {
        "metric"        :       "fps_per_GB",
        "descending"    :       True
    },
    {
        "metric"        :       "mIoU_per_spf",
        "descending"    :       True
    },
    {
        "metric"        :       "Inf time (fps)",
        "descending"    :       True
    },
    {
        "metric"        :       "mIoU",
        "descending"    :       True
    }
]

# model_dict_list = add_n_params_to_model_dicts(model_dict_list=model_dict_list)

model_dict_list = remove_too_big_models(model_dict_list=model_dict_list) 

# print(f"len model dict list (pre unique): {len(model_dict_list)}")
# for model_dict in model_dict_list:
#     print_trimmed_model_dict(model_dict=model_dict)
    
model_dict_list = get_unique_model_list(model_dict_list=model_dict_list)

# print(f"len model dict list (unique): {len(model_dict_list)}")
# for model_dict in model_dict_list:
#     print_trimmed_model_dict(model_dict=model_dict)
    

selected_model_dict_list = []
n_items = 5 # TODO maybe lower, look at result
n_items_provisional = 20
n_total_models = 0
for dataset_name in TrimData.accepted_dataset_list:
    print(f"Dataset: {dataset_name}")
    # Use a bigger provisional selection in order for more generalistic
    #  model to be added
    sorted_by_metric_dict = get_sorted_by_metric_for_dataset(
        dataset_name=dataset_name, 
        model_dict_list=model_dict_list,
        metric_param_list=metric_param_list
    )
    top_n_metric_dict = get_top_n_metric_dict(
        sorted_by_metric_dict=sorted_by_metric_dict,
        n_items=n_items_provisional
    )
    score_list = get_score_list_dataset(top_n_metric_dict=top_n_metric_dict)
    # finer selection
    for metric, top_n in top_n_metric_dict.items():
        top_n_metric_dict[metric] = top_n[:n_items] 
        
    print_n_top_items(top_n_metric_dict=top_n_metric_dict)
    
    sorted_score_list = sorted_score(score_list=score_list, n_items=n_items)
    print_sorted_score(sorted_score_list=sorted_score_list)
    
    
    
    for metric, top_n in top_n_metric_dict.items():
        selected_model_dict_list = select_configs(
            selected_model_dict_list=selected_model_dict_list,
            provisonal_list=top_n
        )
        n_total_models += len(top_n)
    
    selected_model_dict_list = select_configs(
        selected_model_dict_list=selected_model_dict_list,
        provisonal_list=sorted_score_list
    )
    n_total_models += len(sorted_score_list)
    
    
    print(f"n_total_models : {n_total_models}")
    print(f"len selection : {len(selected_model_dict_list)}")
    
    
    print('#' * 80)

# for model_dict in sel

write_selected_files(selected_model_dict_list=selected_model_dict_list)

print("SELECTED MODELS:")
for model_dict in selected_model_dict_list:
    for key, val in model_dict.items():
        print(f"{key} : {val}")
    print()