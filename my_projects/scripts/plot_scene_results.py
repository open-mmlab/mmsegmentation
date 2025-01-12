from matplotlib import pyplot as plt 
import numpy as np
import json 
import os
import argparse
import plotting_utils as p_utils




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scene_results_json_path', 
        '-load',
        type=str,
        help='directory path of scene results',
        default='my_projects/ablation_tests/test_results/data/scene_results.json'
    )
    parser.add_argument(
        '--save_dir_path',
        '-dir',
        type=str,
        default="my_projects/images_plots/ablation"
    )
    
    parser.add_argument(
        '--global_plot',
        '-gp',
        action='store_true'
    )
    parser.add_argument(
        '--per_model',
        '-pm',
        action='store_true'
    )
    parser.add_argument(
        '--single_model',
        '-sm',
        action='store_true'
    )
    parser.add_argument(
        '--show',
        action='store_true'
    )
    parser.add_argument(
        '--dont_save',
        action='store_true'
    )
    # parser.add_argument(
    #     '--average_test_results_path',
    #     '-av',
    #     type=str,
    #     default=None
    # )
    parser.add_argument(
        '--include_av',
        '-incl',
        action='store_true'
    )
    parser.add_argument(
        '--plot_sep',
        '-sep',
        action='store_true'
    )
    
    
      
    args = parser.parse_args()
    if args.single_model:
        args.global_plot = False
        args.per_model = False
    # if args.include_av and args.average_test_results_path is None:
    #     args.average_test_results_path = "my_projects/test_results/arid20_cat/data/arid20_cat_performance_results.json"
    return args


def get_model_dir_name(scene__data_path: str):
    idx_ = -2 if scene__data_path[-1] == '/' else -1
    return scene__data_path.split("/")[idx_] 
 
def get_data_dict(json_path):
    data = None
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def global_data_grouped_by_scene(
    data_dict,
    metric = "mIoU",
    include_av = False
):
    grouped_dict = {
        "floor"     :       [],
        "table"     :       [],
        "bottom"    :       [],
        "top"       :       []
    }
    if include_av:
        grouped_dict['overall'] = []
    for model_name, model_data in data_dict.items():
        for scene_name, scene_data in model_data.items():
            if scene_name not in grouped_dict.keys():
                continue
            grouped_dict[scene_name].append(
                round(scene_data[metric], 1)
            ) 
    
    return grouped_dict

def seperate_by_scene_type(
    grouped_by_scene,
    scene_type_sets = {
        "viewpoint"     :       ["bottom", "top"],
        "place"         :       ["floor", "table"]
    }
):
    seperated = {}
    for scene_type, scene_type_names in scene_type_sets.items():
        seperated[scene_type] = {
            scene_name : scene_data 
                for scene_name, scene_data in grouped_by_scene.items()
                    if scene_name in scene_type_names
        }
    return seperated
    
                    

def global_plot(
    data_dict,
    save_dir_path,
    save_file_name = "global",
    save = True,
    show = False,
    metric = "mIoU",
    include_av = False,
    plot_sep = False
):
    
    models = tuple(
        p_utils.map_model_name(model_name=model_name) 
            for model_name in data_dict.keys()
    ) 
    data_dict = fix_metric_keys_global_data_dict(data_dict=data_dict)
    grouped_by_scene = global_data_grouped_by_scene(
        data_dict=data_dict,
        metric=metric,
        include_av=include_av
    )
    param_dict = p_utils.scenes_plot_global
    seperated_by_scene_type = {
        "all"   :   grouped_by_scene
    }
   
    if plot_sep:
        seperated_by_scene_type = seperate_by_scene_type(
            grouped_by_scene=grouped_by_scene
        )
        param_dict = p_utils.scenes_plot_global_per_scene_type
    for scene_type, scene_data in seperated_by_scene_type.items():
        plot_global_plot(
            grouped_by_scene=scene_data,
            models=models,
            save_dir_path=save_dir_path,
            save_file_name=f"{save_file_name}_{scene_type}",
            save=save,
            show=show,
            metric=metric,
            param_dict=param_dict
        )

def plot_global_plot(
    grouped_by_scene,
    models,
    save_dir_path,
    save_file_name = "global",
    save = True,
    show = False,
    metric = "mIoU",
    param_dict=p_utils.scenes_plot_global
):
    p_utils.set_params(param_dict=param_dict)
    x_axis = np.arange(len(models))
    bar_width = 0.96 / len(grouped_by_scene.keys())
    relative_positions = get_relative_positions(
        grouped_dict=grouped_by_scene
    )
    for scene_idx, scene_item in enumerate(grouped_by_scene.items()):
        (scene_name, scene_value) = scene_item
        offset = (bar_width) * relative_positions[scene_idx]
        rects = plt.bar(
            x_axis + offset, 
            scene_value, 
            bar_width,
            label=scene_name
        )
        plt.bar_label(rects, padding=3, fontsize=15)
    
    plt.ylabel(metric)
    plt.yticks(np.arange(0, 110, 10))
    plt.ylim(48, 102)
    plt.xticks(x_axis, models)
    plt.xlabel("model name")
    plt.legend(
        bbox_to_anchor=(0, 1.0, 1, 0.2),
        ncol=3,
        mode="expand"
    )
    
    plt.tight_layout()
    if save:
        if 'png' not in save_file_name:
            save_file_name = f"{save_file_name}.png"
        save_path = os.path.join(
            save_dir_path,
            save_file_name
        )
        print(f"saved fig in: {save_path}")
        plt.savefig(fname=save_path, format='png')  
    if show:
        plt.show()
    plt.clf()
    p_utils.set_params()

def plot_all_models(
    data_dict,
    save_dir_path,
    save = True,
    show = False,
    include_av = False
):
    for model_name, model_data_dict in data_dict.items():
        model_plot(
            model_data_dict=model_data_dict,
            model_name=model_name,
            save_dir_path=save_dir_path,
            save=save,
            show=show,
            include_av=include_av
        )

def fix_keys_metric_data_dict(metric_data_dict):
    fixed_metric_keys = {}
    for metric_name, metric_value in metric_data_dict.items():
        if  "mPr@" in metric_name:
            fixed_metric_keys[metric_name.split(".")[0]] = metric_value
        else:
            fixed_metric_keys[metric_name] = metric_value
    return fixed_metric_keys

def fix_metric_keys_model_data_dict(model_data_dict):
    fixed_metric_keys = {}
    for scene_name, scene_data in model_data_dict.items(): 
        fixed_metric_keys[scene_name] = fix_keys_metric_data_dict(
            metric_data_dict=scene_data
        )
    return fixed_metric_keys

def fix_metric_keys_global_data_dict(data_dict):
    fixed_metric_keys = {}
    for model_name, model_data_dict in data_dict.items():
        fixed_metric_keys[model_name] = fix_metric_keys_model_data_dict(
            model_data_dict=model_data_dict
        )
    return fixed_metric_keys
    

def model_data_grouped_by_metric(
    model_data_dict,
    model_name,
    include_av = False
):
    grouped_dict = {
        "mPr@50"        :       [],
        "mPr@60"        :       [],
        "mPr@70"        :       [],
        "mPr@80"        :       [],
        "mPr@90"        :       [],
        "mIoU"          :       []
    }
    for scene_name, scene_data in model_data_dict.items():
        if not include_av and scene_name == "overall":
            continue
        for metric_name, value in scene_data.items():
            if metric_name not in grouped_dict.keys():
                continue
            grouped_dict[metric_name].append(
                round(value, 1)
            )

        
    return grouped_dict

def get_relative_positions(
    grouped_dict,
    step_size = 1
):
    n_bars = len(list(grouped_dict.keys()))
    # mid = 0 if n_bars % 2 == 0 else -1
    mid = 0
    
    min_pos = mid - (n_bars/2 - step_size/2)
    max_pos = mid + (n_bars/2 - step_size/2)
    
    relative_positions = np.arange(min_pos, max_pos + step_size, step_size)
    return relative_positions

def model_plot(
    model_data_dict,
    model_name,
    save_dir_path,
    save = True,
    show = False,
    include_av = False
):
    
    model_data_dict = fix_metric_keys_model_data_dict(
        model_data_dict=model_data_dict
    )
    
    grouped_by_metric = model_data_grouped_by_metric(
        model_data_dict=model_data_dict,
        model_name=model_name,
        include_av=include_av
    )
    
    p_utils.set_params(
        param_dict=p_utils.scenes_plot_model
    )
    
    scenes = list(model_data_dict.keys()) 
    if not include_av and "overall" in scenes:
        scenes.remove("overall")
    scenes = tuple(scenes)
        
    x_axis = np.arange(len(scenes))
    
    bar_width = 0.95 / len(grouped_by_metric.keys())
    
    relative_positions = get_relative_positions(
        grouped_dict=grouped_by_metric
    )
    for item_idx, metric_item in enumerate(grouped_by_metric.items()):
        (metric_name, metric_value) = metric_item
        offset = (bar_width) * relative_positions[item_idx]
        metric_name = p_utils.map_metric_key_plot(metric_name)
        rects = plt.bar(
            x_axis + offset, 
            metric_value, 
            bar_width,
            label=metric_name
        )
        plt.bar_label(rects, padding=3, fontsize=15)
    
    plt.ylabel("score (%)")
    plt.yticks(np.arange(0, 110, 10))
    plt.ylim(0, 108)
    plt.xticks(x_axis, scenes)
    plt.xlabel("scene type")
    plt.legend(
        bbox_to_anchor=(0, 1.0, 1, 0.2),
        ncol=len(grouped_by_metric.keys()),
        mode="expand"
    )
    
    plt.tight_layout()
    if save:
        save_path = os.path.join(
            save_dir_path,
            f"{model_name}_scenes.png"
        )
        print(f"saved fig in: {save_path}")
        plt.savefig(fname=save_path, format='png')  
    if show:
        plt.show()
    plt.clf()
    p_utils.set_params()
    
    
        
def main():
    args = parse_args()
    p_utils.set_params()
    data_dict = get_data_dict(json_path=args.scene_results_json_path)
             
    if args.single_model:
        model_name = get_model_dir_name(
            scene__data_path=args.scene_results_json_path
        ).split("_")[0]
        model_plot(
            model_data_dict=data_dict,
            model_name=model_name,
            save_dir_path=args.save_dir_path,
            save=(not args.dont_save),
            show=args.show
        )
        
        
    if args.global_plot:
        global_plot(
            data_dict=data_dict,
            save_dir_path=args.save_dir_path,
            save_file_name="global",
            save=(not args.dont_save),
            show=args.show,
            include_av=args.include_av,
            plot_sep=args.plot_sep
        )     
    if args.per_model:
        plot_all_models(
            data_dict=data_dict,
            save_dir_path=args.save_dir_path,
            save=(not args.dont_save),
            show=args.show,
            include_av=args.include_av
            
        )
    p_utils.reset_params()

if __name__ == '__main__':
    main()
