from matplotlib import pyplot as plt 
import numpy as np
import json 
import os
import argparse
import my_projects.scripts.plotting_utils as p_utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--clutter_path', 
        type=str,
        default="my_projects/test_results/arid20_cat/",
        help='directory path of clutter experiment'
    )
    parser.add_argument(
        "--save_dir_path",
        type=str,
        default="my_projects/images_plots/clutter"
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
    
    args = parser.parse_args()
    
    return args


def get_data_dict(json_path):
    data = None
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def reorganize_clutter_dict(clutter_data_dict):
    metrics_names = [
        "mPr@50", "mPr@60", "mPr@70", 
        "mPr@80", "mPr@90", "mIoU"
    ]
    cl_dict_new = {
        metric_name : np.zeros(len(clutter_data_dict))
            for metric_name in metrics_names
    }
    for n_objects, data_dict in clutter_data_dict.items():
        for metric_name, metric_val in data_dict.items():
            # TODO temp split due to wrong notation old tests
            metric_name = p_utils.map_metric_key(key=metric_name)
            if metric_name not in metrics_names:
                continue
            cl_dict_new[metric_name][int(n_objects) - 1] = metric_val
    return cl_dict_new

def make_plot(
    clutter_data_dict,
    save_path, 
    xlabel="n objects",
    ylabel="score (%)"
):
    x_axis = [
        int(n_objects + 1) for n_objects in range(
            len(
                clutter_data_dict[list(clutter_data_dict.keys())[0]]
            )
        )
    ]
    y_axis = np.arange(0, 110, 10)
    legend_handles = []
    # important moments 1-9 separated, 10-16 phys touch, 17-20 stacked
    for data_point in [9, 16, 20]:
        plt.axvline(data_point, color='k', ls ='dotted', linewidth=2.5)
    for metric_name, data in clutter_data_dict.items():
        metric_name = p_utils.map_metric_key_plot(metric_name)
        handle, = plt.plot(
                x_axis, 
                data,
                label=metric_name
        )
        legend_handles.append(
            handle
        )
        # legend_labels.append(p_utils.fix_metric_name(metric_name))
    if ".png" not in save_path:
        save_path = f"{save_path}.png"
    plt.legend(
        handles=legend_handles,
        bbox_to_anchor=(0, 0.98, 1, 0.25),
        ncol=3,
        mode="expand"
    )
    plt.xticks(x_axis)
    plt.yticks(y_axis)
    plt.ylim(48, 101)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(fname=save_path, format='png')
    plt.clf()

def get_clutter_dict(clutter_path: str):
    for file_name in os.listdir(clutter_path):
        if ".json" in file_name:
            clutter_dict = get_data_dict(
                json_path=os.path.join(
                    clutter_path,
                    file_name
                )
            )
            clutter_dict = reorganize_clutter_dict(
                clutter_data_dict=clutter_dict
            )
            return clutter_dict

def plot_clutter_per_model(
    clutter_dict, save_path
):
    
    make_plot(
        clutter_data_dict=clutter_dict,
        save_path=save_path
    )

def global_plot(args):
    dataset_path = args.clutter_path
    global_miou_dict = {}
    for model_dir in os.listdir(dataset_path):
        if model_dir == "data":
            continue
        model_path = os.path.join(dataset_path, model_dir)
        if os.path.isfile(model_path):
            continue
        clutter_path = os.path.join(
            model_path,
            "clutter"
        )
        clutter_dict = get_clutter_dict(
            clutter_path=clutter_path
        )
        
        model_name = model_dir.split("_")[0]
        if args.per_model:
            plot_clutter_per_model(
                clutter_dict=clutter_dict,
                save_path=os.path.join(
                    args.save_dir_path,
                    f"{model_name}_clutter.png"
                )
            )
        model_name_ = p_utils.map_model_name(model_name=model_name)
        global_miou_dict[model_name_] = clutter_dict["mIoU"]
    save_path = os.path.join(args.save_dir_path, f"global_clutter")
    make_plot(
        clutter_data_dict=global_miou_dict,
        save_path=save_path,
        xlabel="n objects", ylabel="mIoU"
    )
 
def main():
    args = parse_args()
    p_utils.set_params(param_dict=p_utils.clutter_plot)
    if args.global_plot:
        global_plot(args=args)         
    else:
        clutter_dict = get_clutter_dict(clutter_path=args.clutter_path)
        print(clutter_dict)
        plot_clutter_per_model(
            clutter_dict=clutter_dict,
            save_path=args.save_dir_path
        )
    p_utils.reset_params()

if __name__ == '__main__':
    main()
