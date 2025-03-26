import os
import json
import argparse
from copy import deepcopy
from my_projects.conversion_tests.converters.conversion_dicts import (
    get_missing_class_names
)
from my_projects.scripts.generate_latex_tables import (
    per_label_per_model as gen_latex,
    load_json_file as load_json,
    save_dict_as_json as save_json
)
from my_projects.scripts.plot_per_label_results import (
    make_class_hist,
    save_class_hist_plot
)

from my_projects.scripts.plotting_utils import map_dataset_name
from my_projects.scripts.generate_json_all_test_results import (
    generate_confusion_dict_dataset as gen_new_conf_dict
)

from my_projects.scripts.clean_json_for_reporting import fix_jsons

import subprocess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'conversion_path',
        type=str
    )
    
    parser.add_argument(
        '--all',
        '-a',
        action='store_true'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default=None
    )
    parser.add_argument(
        '--target',
        type=str,
        default=None
    )
    
    parser.add_argument(
        '--gen_table',
        '-gt',
        action='store_true'
    )
    parser.add_argument(
        '--gen_plot',
        '-gp',
        action='store_true'
    )
    parser.add_argument(
        '--gen_confusion_mat',
        '-gcm',
        action='store_true'
    )

    parser.add_argument(
        '--confusion_mat_file_name',
        '-cmfn',
        type=str,
        default=None
    )
    parser.add_argument(
        '--per_label_file_name',
        '-plfn',
        type=str,
        default=None
    )
    parser.add_argument(
        '--global_results_file_name',
        '-grfn',
        type=str,
        default=None
    )
    
    args = parser.parse_args()
    if args.all:
        return args
    if args.conversion_path[-1] == "/":
        args.conversion_path = args.conversion_path[:-1]
        
    _, conversion_name = os.path.split(args.conversion_path)
        
    if args.source is None or args.target is None:
        
        [args.source, args.target] = conversion_name.upper().split("2")
    
    if not (args.gen_table or args.gen_plot or args.gen_confusion_mat):
        args.gen_table = True
        args.gen_plot = True
        args.gen_confusion_mat = True
    
    if args.confusion_mat_file_name is None:
        data_dir_path = os.path.join(
            args.conversion_path,
            "data"
            )
        args.confusion_mat_file_name = [
            file_name for file_name in os.listdir(data_dir_path)
                if "confusion_top_5.json" in file_name
        ][0]
    if args.per_label_file_name is None:
        args.per_label_file_name = f"{conversion_name}_per_label_results.json"
    if args.global_results_file_name is None:
        args.global_results_file_name = f"{conversion_name}_results.json"
    
    return args

# result dict is any dict with class names as keys
def mark_unkown_labels(
    label_data: dict, 
    unknown_labels: list
) -> dict:
    fixed_dict = {}
    for class_label, value in label_data.items():
        if class_label in unknown_labels:
            class_label = f"*{class_label}"
        fixed_dict[class_label] = value
    return fixed_dict


# result dict is any dict with class names as keys
def remove_unknown_labels(
    label_data: dict, 
    unknown_labels: list
) -> dict:
    fixed_dict = {}
    for class_label, value in label_data.items():
        if class_label in unknown_labels:
            continue
        fixed_dict[class_label] = value
    return fixed_dict


def make_marked_confusion_matrix(
    args,
    conf_matrix: dict,
    unkown_labels: list,
    save_app = ""
):
    new_matrix = deepcopy(conf_matrix)
    for model_name, conf_list in new_matrix.items():
        for conf_item in conf_list:
            for key, value in conf_item.items():
                if value in unkown_labels:
                    conf_item[key] = f"*{value}"
    
    save_file_name = args.confusion_mat_file_name.replace(
        ".json",
        f"{save_app}.json"
    )
            
    save_path = os.path.join(
        args.conversion_path,
        "data",
        save_file_name   
    )
    save_json(
        data_dict=new_matrix,
        dump_file_path=save_path
    )   
    return new_matrix

def make_marked_per_label_results(
    args,
    global_per_label_results: dict,
    unknown_labels: list
):
    fixed_label_results = {}
    for model_name, model_label_data in global_per_label_results.items():
        fixed_model_data = mark_unkown_labels(
            label_data=model_label_data,
            unknown_labels=unknown_labels
        )
        fixed_label_results[model_name] = fixed_model_data
    return fixed_label_results

def plot_per_class_data(
    args,
    fixed_per_label_results: dict,
    save_app = ""
):
    
    for model_name, model_label_data in fixed_per_label_results.items(): 
        model_class_hist = make_class_hist(
            model_data_dict=model_label_data,
            metric="IoU"
        )
        model_plot_save_path = os.path.join(
            args.conversion_path,
            "data",
            f"{model_name.lower()}_per_label_hist{save_app}"
        )
        save_class_hist_plot(
            class_hist=model_class_hist,
            save_path=model_plot_save_path
        )

def gen_per_class_table(
    args,
    fixed_per_label_results: dict,
    save_app = ""
):
    table_file_path = os.path.join(
        args.conversion_path,
        "data",
        f"latex_per_label_table{save_app}.txt"
    )
    source = map_dataset_name(ds_name=args.source)
    target = map_dataset_name(ds_name=args.target)
    dataset_name = f"{source} to {target}"
    gen_latex(
        label_results_dict=fixed_per_label_results,
        dataset_name=dataset_name,
        file_path=table_file_path
    )
    
    
def get_metrics(global_per_label_results):
    for model_name, model_label_data in global_per_label_results.items(): 
        for label, metrics in model_label_data.items():
            return list(metrics.keys())   
    

def fix_global_data(
    args,
    global_per_label_results: dict,
    unknown_labels: list,
    save_app = "",
    n_decimals = 1
):
    metrics = get_metrics(global_per_label_results=global_per_label_results)
    global_average_results = {
        model_name  :   {
            metric :   0 for metric in metrics
        } for model_name in global_per_label_results.keys()
    }
    for model_name, model_label_data in global_per_label_results.items():
        fixed_model_label_data = remove_unknown_labels(
            label_data=model_label_data,
            unknown_labels=unknown_labels
        )
        # sum all metrics for this model
        for label, metric_dict in fixed_model_label_data.items():
            for metric, metric_value in metric_dict.items():
                global_average_results[model_name][metric] += metric_value
        # average results
        n_labels = len(fixed_model_label_data.items())
        for metric, metric_value in global_average_results[model_name].items():
            av_metric_value = round(metric_value / n_labels, n_decimals)
            global_average_results[model_name][metric] = av_metric_value
            
    save_file_name = args.global_results_file_name.replace(
        ".json",
        f"{save_app}.json"
    )
            
    save_path = os.path.join(
        args.conversion_path,
        "data",
        save_file_name   
    )
    save_json(
        data_dict=global_average_results,
        dump_file_path=save_path
    )

def organise_all(args):
    # in this case the entire test_result dir
    test_results_path = args.conversion_path
    
    for results_dir in os.listdir(test_results_path):
        results_dir_path = os.path.join(
            test_results_path,
            results_dir
        )
        subprocess.call(
            [
                "python",
                "my_projects/conversion_tests/scripts/organise_conversion_data.py",
                results_dir_path
            ]
        )
        
def clean_all(args):
    # in this case the entire test_result dir
    test_results_path = args.conversion_path
    for results_dir in os.listdir(test_results_path):
        results_dir_path = os.path.join(
            test_results_path,
            results_dir
        )
        data_path = os.path.join(
            results_dir_path,
            "data"
        )
        for file_name in os.listdir(data_path):
            if "marked" in file_name:
                os.remove(
                    os.path.join(
                        data_path,
                        file_name
                    )
                )
    return
        
def temp_fix(args):
    # in this case the entire test_result dir
    test_results_path = args.conversion_path
    # res dir being a ds
    for results_dir in os.listdir(test_results_path):
        results_dir_path = os.path.join(
            test_results_path,
            results_dir
        )
        conf_dict = gen_new_conf_dict(
            dataset_path=results_dir_path,
            n_confusion_values=5,
            ignore_background=True
        )
        conf_path = os.path.join(
            results_dir_path,
            "data",
            f"{results_dir}_confusion_top_5.json"
        )
        save_json(
            data_dict=conf_dict,
            dump_file_path=conf_path
        )
        fix_jsons(
            confusion_json_path=conf_path
        )

def main():
    args = parse_args()
    print(args)
    
    # return temp_fix(args=args)
    
    if args.all:
        return organise_all(args=args)
    
    per_label_file_path = os.path.join(
        args.conversion_path,
        "data",
        args.per_label_file_name
    )
    
    confusion_matrix_path = os.path.join(
        args.conversion_path,
        "data",
        args.confusion_mat_file_name
    )
    
    
    global_per_label_results = load_json(
        json_file_path=per_label_file_path
    )
    unknown_labels = get_missing_class_names(
        source_dataset=args.source,
        target_dataset=args.target
    )
    save_app = "_marked" if unknown_labels else ""
    print(f"unknown_labels : {unknown_labels}")
    print(f"source, target: {args.source}, {args.target}")
    
    marked_per_label_results = make_marked_per_label_results(
        args=args,
        global_per_label_results=global_per_label_results,
        unknown_labels=unknown_labels
    )
    if args.gen_table:
        gen_per_class_table(
            args=args,
            fixed_per_label_results=marked_per_label_results,
            save_app=save_app
        )
    if args.gen_plot:
        plot_per_class_data(
            args=args,
            fixed_per_label_results=marked_per_label_results,
            save_app=save_app
        )
    # noting to add
    if not unknown_labels:
        return
    if args.gen_confusion_mat:
        confusion_matrix = load_json(
            json_file_path=confusion_matrix_path
        )
        make_marked_confusion_matrix(
            args=args,
            conf_matrix=confusion_matrix,
            unkown_labels=unknown_labels,
            save_app=save_app
        )
    
    fix_global_data(
        args=args,
        global_per_label_results=global_per_label_results,
        unknown_labels=unknown_labels,
        save_app=save_app
    )
if __name__ == '__main__':
    main()