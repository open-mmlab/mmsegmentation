import json 
import os 
import argparse
from mmengine.fileio import dump

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json data')
    parser.add_argument(
        'json_path',
        type=str,
        help='path of train log in json format')
    parser.add_argument(
        '--top_n',
        type=int,
        default=10
    )
    parser.add_argument(
        '--ignore_background',
        action='store_true'
    )
    parser.add_argument('--save_dir_path', type=str, default=None)
    args = parser.parse_args()
    return args



def get_top_n(dict_list, n = 10, metric ='score', reverse=True):
    # sort_list =  sorted(dict_list, key=itemgetter(metric), reverse=True) 
    sort_list =  sorted(dict_list, key=lambda d : d[metric], reverse=reverse) 
    if len(sort_list) >= n:      
        return sort_list[:n]
    return sort_list


def main():
    args = parse_args()
    json_path = args.json_path
    assert json_path.endswith('.json')
    with open(json_path) as json_file:
        json_data = json.load(json_file)
    
    if args.ignore_background:
        json_data = [
            data_dict for data_dict in json_data
                if "_background_" not in data_dict.values()
        ]
    top_n = get_top_n(
        dict_list=json_data,
        n=args.top_n,
        metric='score'
    )
    print(f"top {args.top_n} confusion values: ")
    for conf_dict in top_n:
        for key, val in conf_dict.items():
            print(f"{key} : {val}")
        print()

    if args.save_dir_path is not None:
        file_path = os.path.join(
            args.save_dir_path,
            f"top_{args.top_n}_confusion.json"
        )
        dump(top_n, file_path, indent=4)
if __name__ == '__main__':
    main()