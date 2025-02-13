import json 
import os
import argparse
import numpy as np
from my_projects.scripts.plotting_utils import (
    map_dataset_name, map_metric_key, map_model_name, map_metric_key_strict
)
map_basic2plot = {
    "mPr@50.0"      :   "mPr@50",
    "mPr@60.0"      :   "mPr@60",
    "mPr@70.0"      :   "mPr@70",
    "mPr@80.0"      :   "mPr@80",
    "mPr@90.0"      :   "mPr@90",
    "mIoU"          :   "mIoU",
    "average_fps"   :   "FPS",
    "average_mem"   :   "Mem (MB)"
}

map_basic2latex = {
    "mPr@50.0"      :    "mPr\textsubscript{50}",
    "mPr@60.0"      :    "mPr\textsubscript{60}",
    "mPr@70.0"      :    "mPr\textsubscript{70}",
    "mPr@80.0"      :    "mPr\textsubscript{80}",
    "mPr@90.0"      :    "mPr\textsubscript{90}",
    "mIoU"          :    "mIoU",
    "average_fps"   :    "FPS",
    "average_mem"   :    "Mem (MB)"
}

map_plot2latex = {
    "mPr@50"      :    "mPr\textsubscript{50}",
    "mPr@60"      :    "mPr\textsubscript{60}",
    "mPr@70"      :    "mPr\textsubscript{70}",
    "mPr@80"      :    "mPr\textsubscript{80}",
    "mPr@90"      :    "mPr\textsubscript{90}",
    "mIoU"        :    "mIoU",
    "FPS"         :    "FPS",
    "Mem (MB)"    :    "Mem (MB)"
}

map_plot2basic = {
    value : key for key, value in map_basic2plot.items()
}
# map_plot2latex 
# for key, val in map_basic2latex.items():
#     print(f'   "{key}"   :    "{val}",')
def save_dict_as_json(data_dict, dump_file_path):
    print(f"saving to : {dump_file_path}")
    with open(dump_file_path, 'w') as dump_file:
        json.dump(data_dict, dump_file, indent=4)
        
def load_json_file(json_file_path: str) -> dict:
    with open(json_file_path, 'r') as file:
        return json.load(file)

def make_clean_performance_json_latex(data_dict, n_decimals = 1):
    clean_dict = {}   
    for model_name, model_data in data_dict.items():
        clean_dict[model_name] = {}
        for key, val in model_data.items():
            clean_key = None
            if key in map_basic2latex.keys():
                clean_key = map_basic2latex[key]
            elif key in map_plot2latex.keys():
                clean_key = map_plot2latex[key]
            
            if clean_key is not None:
                clean_dict[model_name][clean_key] = round(val, n_decimals)
    return clean_dict      

def make_latex_results(test_results_path = "my_projects/test_results"):
    for dataset_dir in os.listdir(test_results_path):
        dataset_dir_path = os.path.join(test_results_path, dataset_dir)
        json_file_path = os.path.join(
            dataset_dir_path,
            "data",
            f"{dataset_dir}_results.json"
        )
        data_dict = load_json_file(
            json_file_path=json_file_path
        )
        latex_dict = make_clean_performance_json_latex(data_dict=data_dict)
        latex_file_path = json_file_path.replace("results.json", "latex.json")
        save_dict_as_json(
            data_dict=latex_dict,
            dump_file_path=latex_file_path
        )

def gen_latex_class_label(class_label):
    latex_label = class_label
    # latex_label =  f"\texttt{{{class_label}}}"
    # latex_label = latex_label.replace("_", "_")
    return latex_label

def make_clean_confusion_json_latex(confusion_data_list):
    clean_list = []
    for conf_item in confusion_data_list:
        clean_list.append(
            {
                "gt label"      :       gen_latex_class_label(
                    class_label=conf_item['gt_label']
                ),
                "pred label"    :       gen_latex_class_label(
                    class_label=conf_item['pred_label']
                ),
                "score"         :       round(conf_item['score'], 1)
            }
        )
    return clean_list


def shrink_n_global_confusions(
    test_results_path = "my_projects/test_results",
    new_size=5,
    new_name = None
):
    for dataset_dir in os.listdir(test_results_path):
        dataset_dir_path = os.path.join(test_results_path, dataset_dir)
        json_file_path = os.path.join(
            dataset_dir_path,
            "data",
            f"{dataset_dir}_confusion.json"
        )
        data_dict = load_json_file(
            json_file_path=json_file_path
        )
        for model_name, confusion_dict_list in data_dict.items():
            confusion_dict_list = sorted(
                confusion_dict_list, 
                key=lambda d : d["score"],
                reverse=True
            )
            confusion_dict_list = confusion_dict_list[:new_size]
            data_dict[model_name] = confusion_dict_list
        save_dict_as_json(
            data_dict=data_dict,
            dump_file_path=json_file_path.replace(
                "_confusion.json",
                "_confusion_5.json"
            )
        )

        
def collect_all_confusion_matrix_data(
    test_results_path = "my_projects/test_results",
    n_decimals = 1
):
    
    for dataset_dir in os.listdir(test_results_path):
        confusion_dict = {}
        dataset_dir_path = os.path.join(test_results_path, dataset_dir)
        for model_dir in os.listdir(dataset_dir_path):
            if model_dir == "data":
                continue
            json_file_path = os.path.join(
                dataset_dir_path,
                model_dir,
                "confusion_matrix/top_10_confusion.json"
            )  
            confusion_data_list = load_json_file(
                json_file_path=json_file_path
            )
            model_name = model_dir.split("_")[0]
            confusion_dict[model_name] = confusion_data_list
        confusion_data_save_path = os.path.join(
            dataset_dir_path,
            "data",
            f"{dataset_dir}_confusion.json"
        )
        save_dict_as_json(
            data_dict=confusion_dict,
            dump_file_path=confusion_data_save_path
        )
            
             
def clean_keys_scene(
    json_file_path, 
    map_dict = map_basic2plot,
    n_decimals = 2
):
    data = load_json_file(json_file_path=json_file_path)  
    clean_dict = {}
    for model_name, model_data in data.items():
        clean_dict[model_name] = {}
        for scene_name, scene_data in model_data.items():  
            clean_dict[model_name][scene_name] = {}
            for metric_name, metric_val in scene_data.items():
                new_val = round(metric_val, n_decimals)
                if metric_name in map_dict.keys():
                    new_key = map_dict[metric_name]
                    clean_dict[model_name][scene_name][new_key] = new_val
                elif metric_name in map_dict.values():
                    clean_dict[model_name][scene_name][metric_name] = new_val
    save_dict_as_json(
        data_dict = clean_dict,
        dump_file_path = json_file_path.replace(".json", "_clean.json")
        
    )

def fix_model_names(data_dict: dict) -> dict:
    new_data = {}
    for model_name, model_data in data_dict.items():
        model_name_ = map_model_name(model_name=model_name)
        new_data[model_name_] = model_data
    return new_data

def fix_metrics_keys_and_vals(
    metric_dict: dict, 
    n_decimals = 1
) -> dict:
    new_dict = {}
    for metric_name, metric_val in metric_dict.items():
        metric_name_ = map_metric_key_strict(key=metric_name)
        if not metric_name_ or np.isnan(metric_val):
            continue
        new_dict[metric_name_] = round(metric_val, n_decimals)
    return new_dict

def fix_confusion_dict(confusion_dict: dict, n_decimals = 1) -> dict:
    new_conf_dict = fix_model_names(data_dict=confusion_dict) 
    for model_name, conf_list in new_conf_dict.items():
        for conf_item in conf_list:
            conf_item["score"] = round(conf_item["score"], n_decimals)
    return new_conf_dict

def fix_results_dict(results_dict: dict, n_decimals = 1) -> dict:
    new_dict = fix_model_names(data_dict=results_dict)
    for model_name, metric_dict in new_dict.items():
        new_dict[model_name] = fix_metrics_keys_and_vals(
            metric_dict=metric_dict,
            n_decimals=n_decimals
        )
    return new_dict

def fix_per_label_results_dict(results_dict: dict, n_decimals = 1) -> dict:
    new_dict = fix_model_names(data_dict=results_dict)
    for model_name, label_dict in new_dict.items():
        for label, metric_dict in label_dict.items():
            new_dict[model_name][label] = fix_metrics_keys_and_vals(
                metric_dict=metric_dict,
                n_decimals=n_decimals
            )
    return new_dict

def gen_default_save_path(source_json_path, save_app):
    return source_json_path.replace(
        ".json",
        f"{save_app}.json"
    )

def fix_jsons(
    confusion_json_path = None,
    global_results_json_path = None,
    per_label_results_json_path = None,
    confusion_json_save_path = None,
    global_results_json_save_path = None,
    per_label_results_json_save_path = None,
    n_decimals = 1,
    save_app = ""
):
    if confusion_json_path is not None:
        confusion_matrix = load_json_file(
            json_file_path=confusion_json_path
        )
        confusion_matrix = fix_confusion_dict(
            confusion_dict=confusion_matrix,
            n_decimals=n_decimals
        )
        if confusion_json_save_path is None:
            confusion_json_save_path = gen_default_save_path(
                source_json_path=confusion_json_path,
                save_app=save_app
            )
        save_dict_as_json(
            data_dict=confusion_matrix,
            dump_file_path=confusion_json_save_path
        )
    
    if global_results_json_path is not None:
        results_dict = load_json_file(
            json_file_path=global_results_json_path
        )
        results_dict = fix_results_dict(
            results_dict=results_dict,
            n_decimals=n_decimals
        )
        if global_results_json_save_path is None:
            global_results_json_save_path = gen_default_save_path(
                source_json_path=global_results_json_path,
                save_app=save_app
            )
        save_dict_as_json(
            data_dict=results_dict,
            dump_file_path=global_results_json_save_path
        )
        
    if per_label_results_json_path is not None:
        results_dict = load_json_file(
            json_file_path=per_label_results_json_path
        )
        results_dict = fix_per_label_results_dict(
            results_dict=results_dict,
            n_decimals=n_decimals
        )
        if per_label_results_json_save_path is None:
            per_label_results_json_save_path = gen_default_save_path(
                source_json_path=per_label_results_json_path,
                save_app=save_app
            )
        save_dict_as_json(
            data_dict=results_dict,
            dump_file_path=per_label_results_json_save_path
        )
    
            
        
        
def main():
    # clean_keys_scene(
    #     json_file_path="my_projects/ablation_tests/test_results/data/scene_results.json"
    # )
    test_results_path = "my_projects/conversion_tests/test_results"
    for results_dir in os.listdir(test_results_path):
        results_dir_path = os.path.join(
            test_results_path,
            results_dir
        )
        confusion_json_path = os.path.join(
            results_dir_path,
            "data",
            f"{results_dir}_confusion_top_5.json"
        )
        global_results_json_path = os.path.join(
            results_dir_path,
            "data",
            f"{results_dir}_results.json"
        )
        per_label_results_json_path = os.path.join(
            results_dir_path,
            "data",
            f"{results_dir}_per_label_results.json"
        )
        fix_jsons(
            confusion_json_path=confusion_json_path,
            global_results_json_path=global_results_json_path,
            per_label_results_json_path=per_label_results_json_path
        )
    

if __name__ == '__main__':
    fix_jsons(
        global_results_json_path="my_projects/test_results/sodhots-c/data/sodhots-c_results.json",
        # confusion_json_path="my_projects/test_results/sodhots-c/data/sodhots-c_confusion_top_5.json"
        # per_label_results_json_path="my_projects/test_results/sodhots-c/data/sodhots-c_per_label_results.json"
    )
    # main()
