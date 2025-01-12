import argparse
import os
from copy import deepcopy
from my_projects.scripts.clean_json_for_reporting import (
    load_json_file,
    save_dict_as_json,
    fix_jsons, 
    gen_default_save_path
)
from my_projects.scripts.plotting_utils import (
    map_metric_key,
    map_metric_key_strict
)

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import my_projects.scripts.plotting_utils as p_utils 

KEYS_OF_INTEREST = [
    "mPr@50",
    "mPr@60",
    "mPr@70",
    "mPr@80",
    "mPr@90",  
    "mIoU",
    "FPS",
    "Mem (MB)" 
]



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results_path',
        '-rp',
        help='dir containing model dirs',
        type=str,
        default="my_projects/test_results"
    )
    parser.add_argument(
        '--save_app',
        '-sa',
        type=str,
        help='save file appendix',
        default=""
    )
    parser.add_argument(
        '--save_path',
        '-sp',
        type=str,
        default="my_projects/images_plots/acc_eff_trade_off/"
    )
    parser.add_argument(
        '--fix_jsons',
        '-fj',
        action='store_true'
    )
    parser.add_argument(
        '--scoring_mode',
        '-sm',
        type=str,
        choices=["RANK", "PROP", "ALL", "NORM"],
        default="ALL"
    )
    
    parser.add_argument(
        '--plot_data',
        '-plt',
        action='store_true'
    )
    parser.add_argument(
        '--plot_th_analysis',
        '-plt_th',
        action='store_true'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true'
    )
    args = parser.parse_args()
    
    return args 

def score_model_on_metrics_by_rank(
    model_score_dict: dict,
    dataset_dict: dict,
    metric: str
) -> dict:
    descending_ = descending_order(metric=metric)
    sort_dict = dict(
        sorted(
            dataset_dict.items(), 
            key=lambda d : d[1][metric],
            reverse=descending_
        ) 
    )
    
    for model_rank, (model_name, _) in enumerate(sort_dict.items()):
        n_models = len(dataset_dict.keys())
        model_score = n_models - model_rank
        model_score_dict[model_name][metric] += model_score
        # print(f"rank, name, score: {model_rank}, {model_name}, {model_data[metric]}")
    return model_score_dict

def score_on_metric_by_proportion(
    model_score_dict: dict,
    dataset_dict: dict,
    metric: str
) -> dict:
    metric_vector = [
        metric_dict[metric] 
            for model_name, metric_dict in dataset_dict.items()
    ]
    metric_mean = np.mean(metric_vector)   
    metric_std = np.std(metric_vector)
    modifier = 1 if descending_order(metric=metric) else -1
    for model_name, metric_dict in dataset_dict.items():
        model_metric_score = metric_dict[metric]
        model_metric_score = modifier * (model_metric_score - metric_mean) 
        model_metric_score /= metric_std
        
        model_score_dict[model_name][metric] += model_metric_score
    return model_score_dict

def score_on_metric_normalized(
    model_score_dict: dict,
    dataset_dict: dict,
    metric: str
) -> dict:
    metric_vector = [
        metric_dict[metric] 
            for model_name, metric_dict in dataset_dict.items()
    ]
    if not descending_order(metric=metric):
        metric_vector = [
            1 / metric_dict[metric] 
                for model_name, metric_dict in dataset_dict.items()
        ]
    metric_total = sum(metric_vector)
    for model_idx, (model_name, metric_dict) in enumerate(dataset_dict.items()):
        model_metric_score = metric_vector[model_idx] / metric_total
        model_score_dict[model_name][metric] += model_metric_score
    return model_score_dict


    
SCORING_MODE = {
    "RANK"      :     score_model_on_metrics_by_rank,
    "PROP"      :     score_on_metric_by_proportion,
    "NORM"      :     score_on_metric_normalized
}

def score_model_on_metric(
    model_score_dict: dict,
    dataset_dict: dict,
    metric: str,
    scoring_mode = "NORM"
) -> dict:
    return SCORING_MODE[scoring_mode](
        model_score_dict=model_score_dict,
        dataset_dict=dataset_dict,
        metric=metric
    )
def compress_precision_scores_per_model(
    global_score_per_model: dict
) -> dict:
    # compressed = {}
    # return compressed
    pass 

def calc_efficiency_scores_per_model(
    global_score_per_model: dict
) -> dict:
    # score_dict = {}
    # return score_dict
    pass

def calc_total_score_with_metric_weights(
    global_score_per_model: dict,
    weights: dict,
    ignore_unweighted = True
) -> dict:
    per_model_total_score = {}
    for model_name, metric_score_dict in global_score_per_model.items():
        per_model_total_score[model_name] = 0
        for metric_name, metric_score in metric_score_dict.items():
            if metric_name in weights.keys():
                weighted_val = metric_score * weights[metric_name]
                per_model_total_score[model_name] += weighted_val
            elif not ignore_unweighted:
                per_model_total_score[model_name] += metric_score
    return per_model_total_score

# returns if higher is better 
def descending_order(metric):
    return metric != "Mem (MB)"

def fix_data_dataset(
    args,
    dataset_dir_name
):
    
    json_results_path = os.path.join(
        args.results_path,
        dataset_dir_name,
        "data",
        f"{dataset_dir_name}_results.json"
    )
    fix_jsons(
        global_results_json_path=json_results_path,
        global_results_json_save_path=gen_default_save_path(
            source_json_path=json_results_path, 
            save_app=args.save_app
        )
    )

def fix_results_jsons_all_datasets(args):
    for dataset_dir_name in os.listdir(args.results_path):
        fix_data_dataset(
            args=args,
            dataset_dir_name=dataset_dir_name
        )
        json_results_path = os.path.join(
            args.results_path,
            dataset_dir_name,
            "data",
            f"{dataset_dir_name}_results.json"
        )
        dataset_results_dict = load_json_file(json_file_path=json_results_path)
        dataset_results_dict = filter_dataset_results_keys(
            dataset_results_dict=dataset_results_dict
        )
        save_path = gen_default_save_path(
            source_json_path=json_results_path,
            save_app=args.save_app
        )
        save_dict_as_json(
            data_dict=dataset_results_dict,
            dump_file_path=save_path
        )

def filter_dataset_results_keys(dataset_results_dict: dict) -> dict:
    new_dict = {}
    for model_name, metric_dict in dataset_results_dict.items():
        new_dict[model_name] = {
            map_metric_key(metric_key) : metric_val 
                for metric_key, metric_val in metric_dict.items()
                    if map_metric_key(metric_key) in KEYS_OF_INTEREST
        }
    return new_dict

def collect_all_datasets_results_jsons(args) -> dict:
    global_dict = {}
    for dataset_dir_name in os.listdir(args.results_path):
        json_results_path = os.path.join(
            args.results_path,
            dataset_dir_name,
            "data",
            f"{dataset_dir_name}_results.json"
        )
        dataset_results_dict = load_json_file(json_file_path=json_results_path)
        global_dict[dataset_dir_name] = dataset_results_dict
    return global_dict

def get_model_score_on_dataset(
    dataset_dict: dict,
    scoring_mode = "RANK"
) -> dict:
    model_score_dict = {
        model_name : {
            metric_name     :   0       for metric_name in KEYS_OF_INTEREST
        } for model_name in dataset_dict.keys()
    }
    for metric in KEYS_OF_INTEREST:
        model_score_dict = score_model_on_metric(
            model_score_dict=model_score_dict,
            dataset_dict=dataset_dict,
            metric=metric,
            scoring_mode=scoring_mode
        )
    return model_score_dict

def merge_model_score_dicts(
    ms_dict_0: dict,
    ms_dict_1: dict = None
) -> dict:
    if not ms_dict_1:
        ms_dict_1 = {
            model_name  :  {
                metric_name     :   0   for metric_name in KEYS_OF_INTEREST
            } for model_name in ms_dict_0.keys()
        }
    new_ms_dict = {}
    model_names = [
        key for key in ms_dict_0.keys()
            if key in ms_dict_1.keys()
    ]
    if (
        len(model_names) < len(ms_dict_0.keys())
        or 
        len(model_names) < len(ms_dict_1.keys())
        or 
        len(ms_dict_0.keys()) != len(ms_dict_1.keys())
    ):
        print("error")
        exit()
    for model_name in model_names:
        new_ms_dict[model_name] = {}
        for metric_name in KEYS_OF_INTEREST:
            val_0 = ms_dict_0[model_name][metric_name]
            val_1 = ms_dict_1[model_name][metric_name]
            new_ms_dict[model_name][metric_name] = val_0 + val_1
    return new_ms_dict



def generate_scatter_data(
    global_dict: dict,
    x_key,
    y_key,
    area_key = None,
    mark_models = True
):
    # TODO add params 
    x_data = []
    y_data = []
    area_data = []
    color_data = []
    for dataset_name, dataset_dict in global_dict.items():
        for model_name, model_data in dataset_dict.items():
            x_data.append(model_data[x_key])  
            y_data.append(model_data[y_key]) 
            if area_key:
                area_data.append(model_data[area_key])
            if mark_models:
                color_data.append(p_utils.MODEL_COLORS[model_name])
    return x_data, y_data, area_data, color_data

def make_handles():
    legend_handles = []
    for model_name, color in p_utils.MODEL_COLORS.items():
        
        legend_handles.append(
                mlines.Line2D(
                    [], [],
                    color=color,
                    label=model_name,
                    linestyle='-',
                    linewidth=6
                )
            )
    return legend_handles

def plot_trade_off_global(
    global_dict: dict,
    x_key,
    y_key,
    area_key = None,
    mark_models = True,
    save_path = None
):
    
    x_data, y_data, area_data, color_data = generate_scatter_data(
        global_dict=global_dict,
        x_key=x_key,
        y_key=y_key,
        area_key=area_key,
        mark_models=mark_models
    )
    
    if not area_data:
        area_data = None
    if not color_data:
        color_data = None
    area_data = np.asarray(area_data) * 2
    p_utils.set_params(param_dict=p_utils.TRADEOFF_PLOT_PARAMS) 
    plt.scatter(
        x=x_data,
        y=y_data,
        s=area_data,
        c=color_data
    )
    if color_data:
        plt.legend(handles=make_handles(),
            bbox_to_anchor=(-0.02, 0.98, 1.05, 0.25),
            ncol=3,
            mode="expand"
        )
    plt.xlabel(xlabel=x_key)
    plt.ylabel(ylabel=y_key)
    plt.xlim(70, 94)
    if save_path:
        print(f"saved fig in : {save_path}")
        plt.savefig(
            save_path,
            dpi=100,
            bbox_inches='tight'
        )
    else:
        plt.show()
    p_utils.reset_params()
            
        
def collect_score_per_model_over_all_datasets(
    global_dict: dict,
    scoring_mode = "RANK"
) -> tuple:
    total_model_score_dict = {}
    per_dataset_score = {}
    for dataset_name, dataset_dict in global_dict.items():
        model_score_dict = get_model_score_on_dataset(
            dataset_dict=dataset_dict,
            scoring_mode=scoring_mode
        )
        per_dataset_score[dataset_name] = model_score_dict
        total_model_score_dict = merge_model_score_dicts(
            ms_dict_0=model_score_dict, 
            ms_dict_1=total_model_score_dict
        )
    return total_model_score_dict, per_dataset_score



def analyse_weights(
    total_model_score_dict: dict,
    labeled_custom_weights: dict,
    verbose: bool = False
):
    model_weight_data = {}
    for label, custom_weights in labeled_custom_weights.items():
        
        
        model_score_global = calc_total_score_with_metric_weights(
            global_score_per_model=total_model_score_dict,
            weights=custom_weights
        )
        model_score_global = dict(
            sorted(
                model_score_global.items(), 
                key=lambda d: d[1],
                reverse=True
            )
        )
        model_weight_data[label] = model_score_global
        
        if verbose:
            
            print(f"\nweights: {label}")
            for key, val in custom_weights.items():
                print(f"{key} : {val}")
            print("\nResults sorted: ")
            for key, val in model_score_global.items():
                print(f"{key} : {val}")
    return model_weight_data   


def random_analysis(
    total_model_score_dict: dict
):
    
    pr_keys = [
        pr_metric for pr_metric in KEYS_OF_INTEREST
            if "mPr" in pr_metric
    ]
    acc_weights = {
        "mIoU"      :       1
    }
    eff_weights = {
        "FPS"       :       1,
        "Mem (MB)"  :       1
    }
    fps_weights = {
        "FPS"       :       1
    }
    mem_weights = {
        "Mem (MB)"  :       1
    }
    sorted_pr_keys = sorted(pr_keys, reverse=True)
    pr_scaled_by_th = {
        pr_key : (len(sorted_pr_keys) - pr_rank) / len(sorted_pr_keys)
            for pr_rank, pr_key in enumerate(sorted_pr_keys)
    }
    # normalize pr
    pr_scaled_by_th = {
        key     :   val / sum(pr_scaled_by_th.values()) 
            for key, val in pr_scaled_by_th.items()
    }
   
    
    custom_weights = {}
    for key, val in pr_scaled_by_th.items():
        custom_weights[key] = val
    for key, val in eff_weights.items():
        custom_weights[key] = 1 
    for key, val in acc_weights.items():
        custom_weights[key] = 1 
    

    labeled_weights = {
        "Global"            :       custom_weights,
        "Efficiency"        :       eff_weights,
        "FPS"               :       fps_weights,
        "Memory"            :       mem_weights,
        "Precision"         :       pr_scaled_by_th,
        "Accuracy"          :       acc_weights
    }
    
    # present Props:
    labeled_scores = analyse_weights(
        total_model_score_dict=total_model_score_dict,
        labeled_custom_weights=labeled_weights,
        verbose=True
    )
    
def av_results(
    total_model_score_dict: dict,
    n_measurements: int
):
    for model_name, metric_score_dict in total_model_score_dict.items():
        for metric_name, metric_score in metric_score_dict.items():
            metric_score_dict[metric_name] /= n_measurements
            
    
def threshold_analyses(
    total_model_score_dict: dict,
    scoring_mode: str,
    plot_th_analysis: bool = False,
    verbose: bool = False,
    save_path_dir = "my_projects/images_plots/acc_eff_trade_off"
):
    if verbose:
        print(f"{'#' * 80}\n             THRESHOLD_ANALYSES\n")
        
    pr_keys = [
        pr_metric for pr_metric in KEYS_OF_INTEREST
            if "mPr" in pr_metric
    ]
    acc_weights = {
        "mIoU"      :       1
    }
    eff_weights = {
        "FPS"       :       1,
        "Mem (MB)"  :       1
    }
    
    sorted_pr_keys = sorted(pr_keys, reverse=True)
    pr_scaled_by_th = {
        pr_key : (len(sorted_pr_keys) - pr_rank) / len(sorted_pr_keys)
            for pr_rank, pr_key in enumerate(sorted_pr_keys)
    }
    # normalize pr
    pr_scaled_by_th = {
        key     :   val / sum(pr_scaled_by_th.values()) 
            for key, val in pr_scaled_by_th.items()
    }
   
    
    default_weights = {}
    for key, val in pr_scaled_by_th.items():
        default_weights[key] = val
    for key, val in eff_weights.items():
        default_weights[key] = 1 
    for key, val in acc_weights.items():
        default_weights[key] = 1 
    
    th_result_dict = {}
    for metric_key in KEYS_OF_INTEREST:
        if "mPr" in metric_key:
            continue
        factors = [
            fac for fac in np.arange(0, 10, 0.1)
        ]
        labeled_weights = {}
        for fac in factors:
            new_weights = deepcopy(default_weights)
            new_weights[metric_key] = fac
            labeled_weights[fac] = new_weights
        labeled_scores = analyse_weights(
            total_model_score_dict=total_model_score_dict,
            labeled_custom_weights=labeled_weights,
            verbose=False
        )
        th_result_dict[metric_key] = labeled_scores
        if verbose:
            print_labeled_scores(
                metric_key=metric_key,
                default_weights=default_weights,
                labeled_scores=labeled_scores
            )
        if plot_th_analysis:
            plot_threshold_analysis(
                metric_key=metric_key,
                labeled_scores=labeled_scores,
                scoring_mode=scoring_mode,
                save_path_dir=save_path_dir
            )
    # Pr analysis
    factors = [
        fac for fac in np.arange(0, 10, 0.1)
    ]
    labeled_weights = {}
    for fac in factors:
        new_weights = deepcopy(default_weights)
        for pr_key in pr_keys:
            new_weights[pr_key] *= fac
        labeled_weights[fac] = new_weights
    labeled_scores = analyse_weights(
        total_model_score_dict=total_model_score_dict,
        labeled_custom_weights=labeled_weights,
        verbose=False
    )
    th_result_dict["mPr"] = labeled_scores
    if verbose:
        print_labeled_scores(
            metric_key="mPr",
            default_weights=default_weights,
            labeled_scores=labeled_scores
        )
    if plot_th_analysis:
        plot_threshold_analysis(
            metric_key="mPr",
            labeled_scores=labeled_scores,
            scoring_mode=scoring_mode,
            save_path_dir=save_path_dir
        )
    return th_result_dict

def plot_threshold_analysis_compact(
    th_results_dict,
    save_path_dir = "my_projects/images_plots/acc_eff_trade_off"
):
    p_utils.set_params(param_dict=p_utils.PRED_VISUALIZATION_FIGURE_PARAMS_3ROWS)
    figure = plt.figure()
    subplots = figure.subplots(
        nrows=3, 
        ncols=4, 
        gridspec_kw = {
            'wspace':0.01, 
            'hspace':0.001
        }
    )
    y_lims_row = {
        0   :   (0, 50),
        1   :   (-10, 10),
        2   :   (0, 3)
    }
    score_type_row = {
        0   :   "rank",
        1   :   "proportional",
        2   :   "normalized"
    }
    for row_idx, (mode, th_results) in enumerate(th_results_dict.items()):
        for col_idx, (metric_key, labeled_scores) in enumerate(th_results.items()):
            axes = subplots[row_idx][col_idx]
            x_axis = [
                float(factor) for factor in list(labeled_scores.keys())
            ]
            model_score_lists_dict = {
                model_name : [] 
                    for model_name in p_utils.MODEL_COLORS.keys()
            }
            for factor, model_score_dict in labeled_scores.items():
                for model_name, total_score in model_score_dict.items():
                    model_score_lists_dict[model_name].append(total_score)
                    
            for model_name, model_score_list in model_score_lists_dict.items():
                axes.plot(
                    x_axis, 
                    model_score_list, 
                    color=p_utils.MODEL_COLORS[model_name]
                )
                axes.set_ylim(*y_lims_row[row_idx])
                axes.set_xlabel(
                    f"{metric_key} weight",
                    fontsize=16
                )
                axes.set_ylabel(
                    f"score\n{score_type_row[row_idx]}",
                    fontsize=16
                )
                axes.get_yaxis().set_visible(
                    col_idx == 0
                )
                axes.get_xaxis().set_visible(
                    row_idx == 2
                )
    figure.legend(
        handles=make_handles(),
        bbox_to_anchor=(0, 0.94, 1, 0.25),
        ncol=3,
        mode="expand"
    )
    figure.subplots_adjust(hspace=0.01)
    # figure.tight_layout()
    save_path = os.path.join(
        save_path_dir,
        f"all_scores"
    )
    
    print(f"saved fig in : {save_path}")
    plt.savefig(
        save_path,
        dpi=100,
        bbox_inches='tight'
    )
    plt.cla()
    p_utils.reset_params()
    
def plot_threshold_analysis(
    metric_key,
    labeled_scores,
    scoring_mode,
    save_path_dir = "my_projects/images_plots/acc_eff_trade_off"
):
    
    p_utils.set_params(param_dict=p_utils.TRADEOFF_PLOT_PARAMS)
    x_axis = [
        float(factor) for factor in list(labeled_scores.keys())
    ]
    model_score_lists_dict = {
        model_name : [] 
            for model_name in p_utils.MODEL_COLORS.keys()
    }
    for factor, model_score_dict in labeled_scores.items():
        for model_name, total_score in model_score_dict.items():
            model_score_lists_dict[model_name].append(total_score)
            
    for model_name, model_score_list in model_score_lists_dict.items():
        plt.plot(
            x_axis, 
            model_score_list, 
            color=p_utils.MODEL_COLORS[model_name]
        ) 
    plt.legend(
        handles=make_handles(),
        bbox_to_anchor=(0, 0.90, 1, 0.02),
        ncol=3,
        mode="expand")
    save_path = os.path.join(
        save_path_dir,
        f"{scoring_mode}_{metric_key}"
    )
    plt.xlabel(f"{metric_key} weight")
    plt.ylabel("score")
    print(f"saved fig in : {save_path}")
    plt.savefig(
        save_path,
        dpi=100,
        bbox_inches='tight'
    )
    plt.cla()
    p_utils.reset_params()
    

           
def print_labeled_scores(
    metric_key,
    default_weights,
    labeled_scores,
    max_items = 10
):
    print_every = len(labeled_scores.items()) / max_items
    print(f"{'#' * 80}\n {metric_key} comparison")
    print(f"\ndefault weights:")
    for key, val in default_weights.items():
        print(f"{key} : {val}")
    for score_idx, (label, model_score_dict) in enumerate(labeled_scores.items()):
        if score_idx % print_every != 0:
            continue
        print(f"\n{metric_key} weight factor: {label}")
        reverse = descending_order(metric=metric_key)
        model_scores = dict(
            sorted(
                model_score_dict.items(), 
                key=lambda d: d[1],
                reverse=reverse
            )
        )
        for model_name, total_score in model_scores.items():
            print(f"{model_name} : {total_score}")
    



def print_per_model_score(
    total_model_score_dict: dict
):
    print("\nprint per model score")
    for model_name, metric_dict in total_model_score_dict.items():
        print(model_name)
        for metric_name, metric_value in metric_dict.items():
            print(f"{metric_name}   :  {metric_value}")
        print()



def main():
    args = parse_args()
    if args.fix_jsons:
        fix_results_jsons_all_datasets(args=args)
    global_dict = collect_all_datasets_results_jsons(args=args)
    # TODO for now ignore concate dataset
    global_dict = {
        key : val for key, val in global_dict.items() 
            if key != "sodhots-c"
    }
    if args.plot_data:
        save_path = os.path.join(
            args.save_path,
            "global_scatter"
        )
        plot_trade_off_global(
            global_dict=global_dict,
            x_key="mIoU",
            y_key="FPS",
            area_key="Mem (MB)",
            mark_models=True,
            save_path=save_path
        )
    if args.scoring_mode == "ALL":
        th_results_dict = {}
        for mode in SCORING_MODE.keys():
            
            per_model, per_dataset = collect_score_per_model_over_all_datasets(
                global_dict=global_dict,
                scoring_mode=mode
            )
            
            av_results(
                total_model_score_dict=per_model,
                n_measurements=len(per_dataset.keys())
            )
            
            if args.verbose:
                print('#' * 80)
                print(f"scoring_mode: {mode}")
                print_per_model_score(total_model_score_dict=per_model)
                
            # random_analysis(
            #     total_model_score_dict=per_model
            # )
            th_results_dict[mode] = threshold_analyses(
                total_model_score_dict=per_model,
                scoring_mode=mode,
                plot_th_analysis=args.plot_th_analysis,
                verbose=args.verbose,
                save_path_dir=args.save_path
            )
        plot_threshold_analysis_compact(th_results_dict=th_results_dict)
    else:
        per_model, per_dataset = collect_score_per_model_over_all_datasets(
            global_dict=global_dict,
            scoring_mode=args.scoring_mode
        )
        av_results(
            total_model_score_dict=per_model,
            n_measurements=len(per_dataset.keys())
        )
        # random_analysis(
        #     total_model_score_dict=per_model
        # )
        threshold_analyses(
                total_model_score_dict=per_model,
                scoring_mode=args.scoring_mode,
                plot_th_analysis=args.plot_th_analysis,
                verbose=args.verbose,
                save_path_dir=args.save_path
            )
    
    

if __name__ == '__main__':
    main()

    
    