import json 
import os
import argparse
from my_projects.scripts.parse_test_log_table import parse_table


KEYS_OF_INTEREST = {
    "performance"   :   [
        "mPr@50", 
        "mPr@60",
        "mPr@70",
        "mPr@80", 
        "mPr@90", 
        "mIoU"
    ],
    "benchmark"    :   [
        "Mem (MB)",
        "Var Mem",
        "FPS",
        "Var FPS"
    ]
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path',
        '-dsp', 
        type=str,
        default=None,
        help='directory path of dataset results or work dir'
    )
    parser.add_argument(
        '--save_intermediate',
        '-si',
        action='store_true'
    )
    parser.add_argument(
        '--label_results',
        '-lr',
        action='store_true'
    )
    parser.add_argument(
        '--confusion_results',
        '-cr',
        action='store_true'
    )
    parser.add_argument(
        '--general_results',
        '-gr',
        action='store_true'
    )
    parser.add_argument(
        '--n_confusion',
        '-nc',
        type=int,
        default=5
    )
    
    
    args = parser.parse_args()
    # if no specific result is marked, do all by default
    if (
        not args.label_results
        and 
        not args.confusion_results
        and 
        not args.general_results
    ):
        args.label_results = True
        args.confusion_results = True
        args.general_results = True
    return args

def save_dict_as_json(data_dict: dict, dump_file_path: str):
    print(f"saving to : {dump_file_path}")
    with open(dump_file_path, 'w') as dump_file:
        json.dump(data_dict, dump_file, indent=4)
        
def load_json_file(json_file_path: str) -> dict:
    with open(json_file_path, 'r') as file:
        return json.load(file)
    
def is_acc_test(dir_name: str) -> bool:
    file_parts = dir_name.split("_")
    if len(file_parts) < 2:
        return False
    return file_parts[0].isnumeric() and file_parts[1].isnumeric()

def extract_json_file_name_from_dir(dir_path: str) -> str:
    for file_name in os.listdir(dir_path):
        if ".json" in file_name:
            return file_name
    return ""       

def get_data_dict(data_dict: dict, keys_of_interest: list) -> dict:
    if keys_of_interest is None:
        return data_dict
    
    return {
        key : val for key, val in data_dict.items()
            if key in keys_of_interest
    }

def get_data_dict_from_dir(dir_path: str, keys_of_interest: list) -> dict:
    json_file_name = extract_json_file_name_from_dir(
        dir_path=dir_path
    )
    data_dict = load_json_file(
        json_file_path=os.path.join(
            dir_path,
            json_file_name
        )
    )
    return get_data_dict(data_dict=data_dict, keys_of_interest=keys_of_interest)



def merg_dicts(dict_list: list) -> dict:
    new_dict = {}
    for dict_ in dict_list:
        for key, val in dict_.items():
            new_dict[key] = val
    return new_dict   

def get_top_n_confusion_values(
    confusion_dict_list: list,
    n_confusion_values: int
):
    return sorted(
        confusion_dict_list, 
        key=lambda d : d["score"],
        reverse=True
    )[:n_confusion_values]

def generate_confusion_dict_dataset(
    dataset_path: str,
    n_confusion_values: int,
    ignore_background: bool = True,
): 
    confusion_dict = {}
    for model_dir in os.listdir(dataset_path):
        if model_dir == "data":
            continue
        model_name = model_dir.split("_")[0]
        model_results_path = os.path.join(
            dataset_path,
            model_dir
        )
        if os.path.isfile(model_results_path):
            continue
        confusion_matrix_path = os.path.join(
            model_results_path,
            "confusion_matrix",
            "confusion.json"
        )
        if not os.path.exists(confusion_matrix_path):
            print(f"no confusion matrix found at {confusion_matrix_path}")
            continue
        # load dict and get top n
        confusion_matrix = load_json_file(json_file_path=confusion_matrix_path)
        if ignore_background:
        
            confusion_matrix = [
                data_dict for data_dict in confusion_matrix
                    if "_background_" not in data_dict.values()
            ]
        # append top n dict to conf_dict
        confusion_dict[model_name] = get_top_n_confusion_values(
            confusion_dict_list=confusion_matrix,
            n_confusion_values=n_confusion_values
        )
    return confusion_dict

def generate_dict_model_results(model_results_path: str) -> dict:
    model_data_dict = {}
    for dir_name in os.listdir(model_results_path):
        if dir_name == "data":
            continue
        dir_path = os.path.join(model_results_path, dir_name)
        if os.path.isfile(dir_path):
            continue
        dict_ = {}
        if is_acc_test(dir_name=dir_name):
            dict_ = get_data_dict_from_dir(
                dir_path=dir_path,
                keys_of_interest=KEYS_OF_INTEREST['performance']
            )
        elif dir_name == "benchmark":
            dict_ = get_data_dict_from_dir(
                dir_path=dir_path,
                keys_of_interest=KEYS_OF_INTEREST['benchmark']
            )
        else:
            continue  
        model_data_dict = merg_dicts(
            dict_list=[
                model_data_dict,
                dict_
            ]
        )  
    return model_data_dict

def generate_dict_dataset_results(
    dataset_path: str, 
    save_intermediate: bool = False
) -> dict:
    dataset_results_dict = {}
    for model_dir in os.listdir(dataset_path):
        if model_dir == "data":
            continue
        model_name = model_dir.split("_")[0]
        model_results_path = os.path.join(
            dataset_path,
            model_dir
        )
        if os.path.isfile(model_results_path):
            continue
        dataset_results_dict[model_name] = generate_dict_model_results(
            model_results_path=model_results_path
        )
        if save_intermediate:
            save_dict_as_json(
                data_dict=dataset_results_dict[model_name],
                dump_file_path=os.path.join(
                    model_results_path,
                    f"{model_name}_results.json"
                )
            )
    return dataset_results_dict


def generate_dict_per_label_model(
    model_results_path: str
) -> dict:
    
    for dir_name in os.listdir(model_results_path):
        if dir_name == "data":
            continue
        dir_path = os.path.join(model_results_path, dir_name)
        if os.path.isfile(dir_path):
            continue
        dict_ = {}
        if is_acc_test(dir_name=dir_name):
            for file in os.listdir(dir_path):
                if ".log" in file:
                    log_path = os.path.join(
                        dir_path, 
                        file
                    )
                    print(f"log path: {log_path}\n")
                    table_dict = parse_table(
                        log_path=log_path
                    )
                    return table_dict
    return None


def get_per_label_results_dataset(
    dataset_path: str, 
    save_intermediate: bool = False
) -> dict:
    dataset_results_dict = {}
    for model_dir in os.listdir(dataset_path):
        if model_dir == "data":
            continue
        model_name = model_dir.split("_")[0]
        model_results_path = os.path.join(
            dataset_path,
            model_dir
        )
        if os.path.isfile(model_results_path):
            continue
        dataset_results_dict[model_name] = generate_dict_per_label_model(
            model_results_path=model_results_path
        )
        if save_intermediate:
            save_dict_as_json(
                data_dict=dataset_results_dict[model_name],
                dump_file_path=os.path.join(
                    model_results_path,
                    f"{model_name}_label_results.json"
                )
            )
    return dataset_results_dict

def get_model_results_path(model_name: str, dataset_path: str):
    for model_res_name in os.listdir(dataset_path):
        if model_name in model_res_name:
            return os.path.join(
                dataset_path, 
                model_res_name
            )

 

def main():
    args = parse_args()
    if not args.dataset_path:
        return
        
    data_dir_path = os.path.join(
        args.dataset_path,
        "data"
    )
    if not os.path.exists(data_dir_path):
        os.mkdir(data_dir_path)
    _, dataset_name = os.path.split(args.dataset_path)
    if args.dataset_path[-1] == '/' or dataset_name == '':
        _, dataset_name = os.path.split(args.dataset_path[:-1])
    if args.general_results:
        dataset_results_dict = generate_dict_dataset_results(
            dataset_path=args.dataset_path,
            save_intermediate=args.save_intermediate
        )
        
        save_dict_as_json(
            data_dict=dataset_results_dict,
            dump_file_path=os.path.join(
                data_dir_path,
                f"{dataset_name}_results.json"
            )
        )
    if args.confusion_results:
        dataset_confusion_dict = generate_confusion_dict_dataset(
            dataset_path=args.dataset_path,
            n_confusion_values=args.n_confusion
        )
        
        save_dict_as_json(
            data_dict=dataset_confusion_dict,
            dump_file_path=os.path.join(
                data_dir_path,
                f"{dataset_name}_confusion_top_{args.n_confusion}.json"
            )
        )
    if args.label_results:
        label_results = get_per_label_results_dataset(
            dataset_path=args.dataset_path,
            save_intermediate=args.save_intermediate
        )
        
        save_dict_as_json(
            data_dict=label_results,
            dump_file_path=os.path.join(
                data_dir_path,
                f"{dataset_name}_per_label_results.json"
            )
        )
        
                
            
            
        

if __name__ == '__main__':
    main()