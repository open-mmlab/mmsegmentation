import subprocess
import os
import argparse
import plotting_utils as p_utils
import json



MODEL_NAMES = [
    "bisenet",
    "mask2former",
    "maskformer",
    "segformer",
    "segnext"
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--selection_path', 
        type=str,
        help='path of models',
        default='my_projects/best_models/selection_trained/fine_tuning'
    )
    parser.add_argument(
        '--results_path',
        '-rp',
        type=str,
        help='path to store results',
        default='my_projects/training_data'
    )
    parser.add_argument(
        '--reduce_zeros',
        '-rz',
        action='store_true'
    )
    parser.add_argument(
        '--one_plot',
        '-op',
        action='store_true'
    )
    args = parser.parse_args()
    

    return args

def get_json_path(model_dir_path):
    json_path = None
    for dir in os.listdir(model_dir_path):
        dir_path = os.path.join(
            model_dir_path,
            dir
        )
        if os.path.isdir(dir_path):
            json_path = os.path.join(
                dir_path,
                "vis_data",
                "scalars.json"
            )
            if not os.path.exists(json_path):
                json_path = None
            break
    return json_path

def fix_jsons(model_dir_path):
    json_dicts = []
    for dir in os.listdir(model_dir_path):
        dir_path = os.path.join(
            model_dir_path,
            dir
        )
        if os.path.isdir(dir_path):
            if "vis_data" not in os.listdir(dir_path):
                vis_data_path = os.path.join(
                    dir_path, "vis_data"
                )
                os.mkdir(vis_data_path)
            log_file = [
                log_file for log_file in os.listdir(dir_path)
                    if ".log" in log_file
            ][0]
            log_path = os.path.join(dir_path, log_file)
            with open(log_path, 'r') as file_ptr:
                current_train_iter = 0
                for line in file_ptr:
                    if "Iter(train) [" in line:
                        print("iter")
                        current_train_iter = line.split('[', 1)[1].split(
                            ']')[0]
                        
                        current_train_iter = int(
                            current_train_iter.split("/")[0]
                        )
                        print(current_train_iter)
                        # current_train_iter = int(
                        #     re.search(r"\[(\w+\]", line)
                        # )
                    if "Iter(val)" in line:
                        miou = float(line.split("mIoU: ")[-1].split()[0])
                            
                        json_dicts.append(
                            {
                                "step"  : current_train_iter,
                                "mIoU"  : miou
                            }
                        )
            vis_data_path = os.path.join(
                dir_path, "vis_data"
            )
            scalars_path = os.path.join(vis_data_path, "scalars.json")
            with open(scalars_path, 'a') as scal_file:
                for json_dict in json_dicts:
                    json.dump(json_dict, scal_file)
                    scal_file.write("\n")
                            
                             
            

def main():
    args = parse_args()
    # for model_name in MODEL_NAMES:
    #     jsons = {}
    #     for dataset_name in os.listdir(args.selection_path):
    #         dataset_path = os.path.join(args.selection_path, dataset_name)
    #         for model_dir_name in os.listdir(dataset_path):
    #             if model_name in model_dir_name:
    #                 model_dir_path = os.path.join(dataset_path, model_dir_name)
    #                 json_path = get_json_path(model_dir_path=model_dir_path)
    #                 print(f"json_pth = {json_path}")
    #                 if json_path is None:
    #                     print("fix json")
    #                     fix_jsons(model_dir_path=model_dir_path)
                        
    for model_name in MODEL_NAMES:
        jsons = {}
        for dataset_name in os.listdir(args.selection_path):
            dataset_path = os.path.join(args.selection_path, dataset_name)
            for model_dir_name in os.listdir(dataset_path):
                if model_name in model_dir_name:
                    model_dir_path = os.path.join(dataset_path, model_dir_name)
                    json_path = get_json_path(model_dir_path=model_dir_path)
                    if json_path is None:
                        print(f"no json found in {model_dir_path}")
                        continue
                    jsons[dataset_name] = json_path
        call_list = [
            "python",
            "my_projects/scripts/analyze_logs.py"
        ]
        for json_path in jsons.values():
            call_list.append(json_path)
        call_list.append("--keys")
        call_list.append("mIoU")
        # call_list.append("--title")
        # call_list.append(f"{model_name}_learning")
        
        call_list.append("--legend")
        for dataset_name in jsons.keys():
            call_list.append(p_utils.map_dataset_name(dataset_name))
        
        save_path = os.path.join(
            args.results_path,
            f"{model_name}_learning.png"
        )
        call_list.append("--out")
        call_list.append(save_path)
        
        if args.reduce_zeros:
            call_list.append('-rz')
        
        subprocess.call(call_list)
    
if __name__ == '__main__':
    main()