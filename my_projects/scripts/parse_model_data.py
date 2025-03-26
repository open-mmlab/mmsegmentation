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



def is_dataset_name(
    line,
    
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
):

    model_dict_list = []
    for project_name in project_list:
        project_path = os.path.join(config_root_path, project_name)
        project_model_dict_list = get_readme_data(
            project_path=project_path
        )
       
        if project_model_dict_list is None:
            continue
        
        project_model_dict_list = postprocesses_model_dict_list(
            model_dict_list=project_model_dict_list
        )
        model_dict_list += project_model_dict_list
    return model_dict_list

