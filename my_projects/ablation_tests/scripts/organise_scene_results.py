import os
import argparse
import json

def find_json_file(scene_dir_path):
    for root, dirs, files in os.walk(os.path.join(scene_dir_path)):
        json_file = [file for file in files if ".json" in file]
        if json_file:
            return os.path.join(root, json_file[0])
    return None

def fix_keys_scene_dict(
    scene_dict, 
    keys_of_interest = [
        "mPr@50", "mPr@60", "mPr@70",
        "mPr@80", "mPr@90", "mIoU"
    ]
):
    fixed_dict = {}
    for scene_id, scene_data in scene_dict.items():
        fixed_dict[scene_id] = {}
        for metric_key, metric_val in scene_data.items():
            fixed_key = metric_key.split(".")[0]
            if fixed_key in keys_of_interest:
                fixed_dict[scene_id][fixed_key] = metric_val
    return fixed_dict
            

def collect_json_files(model_name, args):
    scene_dict = {}
    model_path = os.path.join(
        args.results_path,
        model_name
    )
    for scene_dir in os.listdir(model_path):
        json_file_path = find_json_file(
            scene_dir_path=os.path.join(
                model_path,
                scene_dir
            )
        )
        if json_file_path is None:
            continue
        with open(json_file_path, 'r') as json_file:
            scene_dict[scene_dir] = json.load(json_file)
    if args.include_av:
        with open(args.average_test_results_path, 'r') as json_file:
            model_key = model_name.split("_")[0]
            scene_dict["overall"] = json.load(json_file)[model_key]
    scene_dict = fix_keys_scene_dict(scene_dict=scene_dict)
    if args.save_intermediate:
        scene_dump_path = os.path.join(model_path, "scene_data.json")
        with open(scene_dump_path, 'w') as dump_file:
            json.dump(scene_dict, dump_file, indent=4)
    return scene_dict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'results_path', 
        type=str
    )
    parser.add_argument(
        '--save_intermediate',
        '-si',
        action='store_true'
    )
    parser.add_argument(
        '--average_test_results_path',
        '-av',
        type=str,
        default=None
    )
    parser.add_argument(
        '--include_av',
        '-incl',
        action='store_true'
    )
      
    args = parser.parse_args()
    if args.include_av and args.average_test_results_path is None:
        args.average_test_results_path = "my_projects/test_results/arid20_cat/data/arid20_cat_performance_results.json"
    return args
def main():
    
    args = parse_args()
    data_dict = {}
    for model_name in os.listdir(args.results_path):
        if model_name == "data":
            continue
        model_path = os.path.join(args.results_path, model_name)
        if not os.path.isdir(model_path):
            continue
        scene_dict = collect_json_files(model_name, args)
        data_dict[model_name.split("_")[0]] = scene_dict
    results_data_file_path = os.path.join(
        args.results_path,
        "scene_results.json"
    )
    with open(results_data_file_path, 'w') as dump_file:
        json.dump(data_dict, dump_file, indent=4)
    
if __name__ == '__main__':
    main()
